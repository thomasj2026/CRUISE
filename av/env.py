"""
CarlaEnv: owns the connection to the CARLA server and everything spawned in it.
"""
import math
import random

import carla

from .sensors import Sensor


class CarlaEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg.get("carla", {})
        self.sync = c.get("sync", True)
        self.follow_spectator = c.get("follow_spectator", True)
        self.rng = random.Random(cfg.get("seed"))

        self.client = carla.Client(c.get("host", "localhost"), int(c.get("port", 2000)))
        self.client.set_timeout(float(c.get("timeout", 10.0)))
        self.world = self._load_world(c.get("town"))
        self.map = self.world.get_map()

        # Remember the server's settings so close() can put them back.
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = self.sync
        settings.fixed_delta_seconds = c.get("fixed_delta_seconds", 0.05) if self.sync else None
        settings.no_rendering_mode = not c.get("render", True)
        self.world.apply_settings(settings)

        traffic = cfg.get("traffic", {})
        self.tm_port = int(traffic.get("tm_port", 8000))
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_synchronous_mode(self.sync)
        if cfg.get("seed") is not None:
            self.tm.set_random_device_seed(int(cfg["seed"]))

        weather = cfg.get("weather")
        if weather:
            self.world.set_weather(getattr(carla.WeatherParameters, weather))

        self.ego = None
        self.sensors = {}
        self.npc_ids = []
        self.frame = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self):
        """(Re)spawn the ego vehicle, its sensors and NPC traffic. Returns the first observation."""
        self._destroy_actors()
        self._spawn_ego()
        self._spawn_sensors()
        self._spawn_traffic()

        # Let the car settle onto the road, then discard spawn-time events.
        for _ in range(10):
            self._tick()
        for s in self.sensors.values():
            if not s.spec.streaming:
                s.clear()
        return self._observe()

    def step(self, control=None):
        """Apply a control (or None to leave control to e.g. autopilot), advance one tick."""
        if control is not None:
            self.ego.apply_control(control)
        self._tick()
        if self.follow_spectator:
            self._update_spectator()
        return self._observe()

    def close(self):
        self._destroy_actors()
        try:
            self.tm.set_synchronous_mode(False)
            self.world.apply_settings(self._original_settings)
        except RuntimeError:
            pass  # server already gone

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_world(self, town):
        world = self.client.get_world()
        if town and not world.get_map().name.endswith(town):
            print(f"[env] Loading {town} ...")
            world = self.client.load_world(town)
        return world

    def _tick(self):
        if self.sync:
            self.frame = self.world.tick()
        else:
            self.frame = self.world.wait_for_tick().frame

    def _spawn_ego(self):
        ego_cfg = self.cfg.get("ego", {})
        bp = self.world.get_blueprint_library().find(ego_cfg.get("blueprint", "vehicle.tesla.model3"))
        bp.set_attribute("role_name", "hero")

        points = self.map.get_spawn_points()
        idx = ego_cfg.get("spawn_point")
        candidates = [points[idx]] if idx is not None else self.rng.sample(points, len(points))
        for pt in candidates:
            self.ego = self.world.try_spawn_actor(bp, pt)
            if self.ego is not None:
                break
        if self.ego is None:
            raise RuntimeError("Could not spawn the ego vehicle (spawn point occupied?)")
        self._tick()

    def _spawn_sensors(self):
        for s_cfg in self.cfg.get("sensors", []):
            s = Sensor(
                self.world, self.ego,
                name=s_cfg["name"], type=s_cfg["type"],
                transform=s_cfg.get("transform"), attributes=s_cfg.get("attributes"),
            )
            self.sensors[s.name] = s

    def _spawn_traffic(self):
        n = int(self.cfg.get("traffic", {}).get("vehicles", 0))
        if n <= 0:
            return
        blueprints = [
            bp for bp in self.world.get_blueprint_library().filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]
        ego_loc = self.ego.get_location()
        points = [p for p in self.map.get_spawn_points() if p.location.distance(ego_loc) > 10.0]
        self.rng.shuffle(points)

        SpawnActor, SetAutopilot, FutureActor = (
            carla.command.SpawnActor, carla.command.SetAutopilot, carla.command.FutureActor)
        batch = [
            SpawnActor(self.rng.choice(blueprints), pt).then(SetAutopilot(FutureActor, True, self.tm_port))
            for pt in points[:n]
        ]
        results = self.client.apply_batch_sync(batch, self.sync)
        self.npc_ids = [r.actor_id for r in results if not r.error]
        print(f"[env] Spawned {len(self.npc_ids)} NPC vehicles")

    def _observe(self):
        t = self.ego.get_transform()
        v = self.ego.get_velocity()
        return {
            "frame": self.frame,
            "timestamp": self.world.get_snapshot().timestamp.elapsed_seconds,
            "ego": {
                "transform": t,
                "x": t.location.x, "y": t.location.y, "z": t.location.z,
                "yaw": t.rotation.yaw,
                "speed": math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),
                "velocity": (v.x, v.y, v.z),
            },
            "sensors": {name: s.get(self.frame, self.sync) for name, s in self.sensors.items()},
        }

    def _update_spectator(self):
        t = self.ego.get_transform()
        fwd = t.get_forward_vector()
        loc = carla.Location(x=t.location.x - 8.0 * fwd.x, y=t.location.y - 8.0 * fwd.y, z=t.location.z + 4.0)
        self.world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-15.0, yaw=t.rotation.yaw)))

    def _destroy_actors(self):
        for s in self.sensors.values():
            s.destroy()
        self.sensors = {}
        if self.npc_ids:
            self.client.apply_batch([carla.command.DestroyActor(i) for i in self.npc_ids])
            self.npc_ids = []
        if self.ego is not None and self.ego.is_alive:
            self.ego.destroy()
        self.ego = None
