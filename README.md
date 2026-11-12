# CARLA AV Framework

A small, modular starting point for autonomous-driving experiments in CARLA.
One YAML file describes the world, the ego car, its sensors, and the driving agent.
Sensors and agents are plug-ins you register with a decorator.

```
carla_av/
├── config.yaml        # everything about a run
├── run.py             # main loop: observe -> agent -> control -> tick
├── requirements.txt
└── av/
    ├── env.py         # CarlaEnv: connect, sync mode, spawn ego/sensors/traffic, reset/step/close
    ├── sensors.py     # sensor registry + converters (raw CARLA data -> numpy)
    ├── agents.py      # agent registry: idle, autopilot, waypoint_follower, lidar_aeb
    ├── display.py     # pygame viewer for any camera or a LiDAR bird's-eye view
    └── config.py      # YAML loader
```

## Setup

1. Download and unzip CARLA 0.9.16 (Windows or Ubuntu).
2. Create a Python environment and install dependencies (the `carla` package version must match the server):
   ```
   pip install -r requirements.txt
   ```
3. Start the server: `CarlaUE4.exe` (Windows) or `./CarlaUE4.sh` (Linux).
   Add `-quality-level=Low` if you want more FPS.

## Run

```
python run.py                                 # config.yaml defaults
python run.py --agent autopilot               # let CARLA drive (good for data collection)
python run.py --agent lidar_aeb               # waypoint follower + LiDAR emergency braking
python run.py --town Town05 --steps 1000
python run.py --no-display
```

Press ESC or close the window to stop. Actors are cleaned up and server settings restored on exit.

## How a step works

```
obs = env.reset()            # spawn everything, return first observation
agent.setup(env)
loop:
    control = agent.run_step(obs)   # carla.VehicleControl, or None for autopilot
    obs = env.step(control)         # apply control, tick the world, collect sensors
```

The simulation runs in **synchronous mode** by default (20 Hz), so every sensor reading in
`obs` belongs to the same simulation frame and the world waits for your code, no matter how
slow your perception model is.

`obs` looks like:

```python
{
  "frame": 1234, "timestamp": 61.7,
  "ego": {"transform": carla.Transform, "x", "y", "z", "yaw", "speed", "velocity"},
  "sensors": {
      "front_cam": np.ndarray (H, W, 3) RGB,
      "lidar":     np.ndarray (N, 4) [x, y, z, intensity],
      "gnss":      {"lat", "lon", "alt"},
      "imu":       {"accel", "gyro", "compass"},
      "collision": [ {"other_actor", "impulse"}, ... ],   # events this tick
      ...
  },
}
```

## Adding or changing sensors

Edit `sensors:` in `config.yaml`. Built-in types: `rgb_camera`, `depth_camera`,
`semantic_camera`, `lidar`, `radar`, `gnss`, `imu`, `collision`, `lane_invasion`.
`attributes` are passed straight to the CARLA blueprint (image size, FOV, LiDAR channels,
noise parameters, etc.).

For a new modality, register a converter in `av/sensors.py`:

```python
@register_sensor("instance_camera", "sensor.camera.instance_segmentation")
def _instance(data):
    return np.frombuffer(data.raw_data, dtype=np.uint8).reshape(data.height, data.width, 4)[:, :, :3].copy()
```

## Adding your own autonomy

Register an agent in `av/agents.py` (or any module imported by `run.py`):

```python
@register_agent("my_agent")
class MyAgent(Agent):
    def __init__(self, target_speed_kmh=20, **params):
        super().__init__(**params)
        self.target = target_speed_kmh / 3.6

    def run_step(self, obs):
        image = obs["sensors"]["front_cam"]
        # perception -> planning -> control
        self.info = {"note": "shown on the HUD"}
        return carla.VehicleControl(throttle=0.3, steer=0.0)
```

Then in `config.yaml`:

```yaml
agent:
  type: my_agent
  params: {target_speed_kmh: 25}
```

`waypoint_follower` is a good base class: it already has a route planner, pure-pursuit steering
and PID speed control, so you can override only the part you're working on
(see `lidar_aeb` for an example that adds a perception check on top).

## Notes

- `waypoint_follower` uses ground-truth pose and the HD map. Replace `obs["ego"]` with your own
  GNSS/IMU/LiDAR localization when you work on that part.
- `render: false` speeds things up but cameras then return black images.
- If you change `fixed_delta_seconds`, set LiDAR `rotation_frequency` to `1 / fixed_delta_seconds`
  so each tick contains one full sweep.
