"""
Autonomy layer.
"""
import math
import random
from collections import deque

import numpy as np

import carla

AGENT_REGISTRY = {}


def register_agent(name):
    def deco(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return deco


def make_agent(agent_cfg):
    agent_cfg = agent_cfg or {}
    name = agent_cfg.get("type", "idle")
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Registered: {sorted(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name](**(agent_cfg.get("params") or {}))


class Agent:
    def __init__(self, **params):
        self.params = params
        self.info = {}          # anything here is shown on the display HUD

    def setup(self, env):
        """Called after every env.reset() with the fresh ego vehicle."""
        self.env = env
        self.vehicle = env.ego
        self.map = env.map

    def run_step(self, obs):
        raise NotImplementedError


# ------------------------------------------------------------------
# Built-in agents
# ------------------------------------------------------------------
@register_agent("idle")
class IdleAgent(Agent):
    """Does nothing. Useful for checking sensors."""
    def run_step(self, obs):
        return carla.VehicleControl(throttle=0.0, brake=0.0)


@register_agent("autopilot")
class AutopilotAgent(Agent):
    """Hands the car to CARLA's Traffic Manager. Handy for collecting sensor data."""
    def setup(self, env):
        super().setup(env)
        self.vehicle.set_autopilot(True, env.tm_port)

    def run_step(self, obs):
        return None


@register_agent("waypoint_follower")
class WaypointFollowerAgent(Agent):
    """
    Baseline modular controller using ground-truth localization and the HD map:
      - route:     random walk along map waypoints
      - lateral:   pure pursuit
      - longitudinal: PID on speed
      - stops at red traffic lights
    Swap any piece out for your own perception / planning / control.
    """

    def __init__(self, target_speed_kmh=30.0, lookahead_gain=0.8, min_lookahead=5.0,
                 kp=0.6, ki=0.05, kd=0.02, waypoint_spacing=2.0, seed=None, **params):
        super().__init__(**params)
        self.target_speed = target_speed_kmh / 3.6
        self.k_ld, self.min_ld = lookahead_gain, min_lookahead
        self.kp, self.ki, self.kd = kp, ki, kd
        self.spacing = waypoint_spacing
        self.rng = random.Random(seed)

    def setup(self, env):
        super().setup(env)
        self.route = deque()
        self._integral = 0.0
        self._prev_err = 0.0
        self.dt = env.cfg.get("carla", {}).get("fixed_delta_seconds", 0.05) or 0.05
        self.wheelbase = 2.9
        wheels = self.vehicle.get_physics_control().wheels
        self.max_steer = math.radians(wheels[0].max_steer_angle) if wheels else math.radians(70.0)

    # --- planning ---------------------------------------------------
    def _extend_route(self, n=60):
        if not self.route:
            loc = self.vehicle.get_location()
            self.route.append(self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving))
        while len(self.route) < n:
            options = self.route[-1].next(self.spacing)
            if not options:
                break
            self.route.append(self.rng.choice(options))

    def _prune_route(self, t):
        yaw = math.radians(t.rotation.yaw)
        while len(self.route) > 1:
            wp = self.route[0].transform.location
            dx, dy = wp.x - t.location.x, wp.y - t.location.y
            ahead = dx * math.cos(yaw) + dy * math.sin(yaw)
            if ahead < 1.0:   # behind or under the car
                self.route.popleft()
            else:
                break

    # --- control ----------------------------------------------------
    def _lateral(self, t, speed):
        ld = max(self.min_ld, self.k_ld * speed)
        target = self.route[-1].transform.location
        for wp in self.route:
            if wp.transform.location.distance(t.location) >= ld:
                target = wp.transform.location
                break
        yaw = math.radians(t.rotation.yaw)
        dx, dy = target.x - t.location.x, target.y - t.location.y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy    # +y = right in CARLA
        alpha = math.atan2(local_y, local_x)
        steer_angle = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        return float(np.clip(steer_angle / self.max_steer, -1.0, 1.0))

    def _longitudinal(self, speed, target_speed):
        err = target_speed - speed
        self._integral = float(np.clip(self._integral + err * self.dt, -10.0, 10.0))
        deriv = (err - self._prev_err) / self.dt
        self._prev_err = err
        u = self.kp * err + self.ki * self._integral + self.kd * deriv
        if u >= 0:
            return float(min(u, 0.75)), 0.0
        return 0.0, float(min(-u, 1.0))

    def _red_light(self):
        return (self.vehicle.is_at_traffic_light()
                and self.vehicle.get_traffic_light_state() == carla.TrafficLightState.Red)

    def run_step(self, obs):
        t, speed = obs["ego"]["transform"], obs["ego"]["speed"]
        self._extend_route()
        self._prune_route(t)

        target_speed = 0.0 if self._red_light() else self.target_speed
        steer = self._lateral(t, speed)
        throttle, brake = self._longitudinal(speed, target_speed)
        if target_speed == 0.0:
            throttle, brake = 0.0, max(brake, 0.5)

        self.info = {"target km/h": round(target_speed * 3.6), "route pts": len(self.route)}
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


@register_agent("lidar_aeb")
class LidarAEBAgent(WaypointFollowerAgent):
    """
    Example of using a sensor in the loop: waypoint follower + automatic
    emergency braking when LiDAR sees points in a corridor ahead of the car.
    """

    def __init__(self, lidar_sensor="lidar", brake_distance=10.0, half_width=1.3,
                 z_min=-2.1, z_max=0.5, **kwargs):
        super().__init__(**kwargs)
        self.lidar_sensor = lidar_sensor
        self.brake_distance = brake_distance
        self.half_width = half_width
        self.z_min, self.z_max = z_min, z_max   # relative to the LiDAR mount; z_min filters out the road

    def run_step(self, obs):
        control = super().run_step(obs)
        pts = obs["sensors"].get(self.lidar_sensor)
        if pts is None or len(pts) == 0:
            return control
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        mask = (x > 2.5) & (x < self.brake_distance) & (np.abs(y) < self.half_width) \
            & (z > self.z_min) & (z < self.z_max)
        n = int(mask.sum())
        self.info["obstacle pts"] = n
        if n > 15:
            self.info["AEB"] = "BRAKING"
            return carla.VehicleControl(throttle=0.0, steer=control.steer, brake=1.0)
        self.info.pop("AEB", None)
        return control
