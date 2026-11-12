"""
Sensor layer
"""
import math
import queue
from dataclasses import dataclass
from typing import Callable

import numpy as np

import carla


@dataclass
class SensorSpec:
    blueprint: str
    convert: Callable
    streaming: bool = True   # True: produces data every tick. False: event-based (collisions etc.)


SENSOR_REGISTRY = {}


def register_sensor(name, blueprint, streaming=True):
    def deco(fn):
        SENSOR_REGISTRY[name] = SensorSpec(blueprint, fn, streaming)
        return fn
    return deco


# ------------------------------------------------------------------
# Converters: raw CARLA measurement -> python / numpy
# ------------------------------------------------------------------
def _bgra(data):
    return np.frombuffer(data.raw_data, dtype=np.uint8).reshape(data.height, data.width, 4)


@register_sensor("rgb_camera", "sensor.camera.rgb")
def _rgb(data):
    """(H, W, 3) uint8 RGB image."""
    return _bgra(data)[:, :, 2::-1].copy()


@register_sensor("depth_camera", "sensor.camera.depth")
def _depth(data):
    """(H, W) float32 depth in meters (0-1000)."""
    arr = _bgra(data).astype(np.float32)
    normalized = (arr[:, :, 2] + arr[:, :, 1] * 256.0 + arr[:, :, 0] * 65536.0) / (256.0 ** 3 - 1)
    return normalized * 1000.0


@register_sensor("semantic_camera", "sensor.camera.semantic_segmentation")
def _semantic(data):
    """(H, W) uint8 class labels (CARLA semantic tags)."""
    return _bgra(data)[:, :, 2].copy()


@register_sensor("lidar", "sensor.lidar.ray_cast")
def _lidar(data):
    """(N, 4) float32 points [x, y, z, intensity] in the sensor frame (x forward)."""
    return np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()


@register_sensor("radar", "sensor.other.radar")
def _radar(data):
    """(N, 4) float32 detections [velocity m/s, azimuth rad, altitude rad, depth m]."""
    return np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()


@register_sensor("gnss", "sensor.other.gnss")
def _gnss(data):
    return {"lat": data.latitude, "lon": data.longitude, "alt": data.altitude}


@register_sensor("imu", "sensor.other.imu")
def _imu(data):
    a, g = data.accelerometer, data.gyroscope
    return {
        "accel": np.array([a.x, a.y, a.z], dtype=np.float32),   # m/s^2
        "gyro": np.array([g.x, g.y, g.z], dtype=np.float32),    # rad/s
        "compass": data.compass,                                # rad, 0 = north
    }


@register_sensor("collision", "sensor.other.collision", streaming=False)
def _collision(data):
    i = data.normal_impulse
    return {
        "other_actor": data.other_actor.type_id,
        "impulse": math.sqrt(i.x ** 2 + i.y ** 2 + i.z ** 2),
    }


@register_sensor("lane_invasion", "sensor.other.lane_invasion", streaming=False)
def _lane_invasion(data):
    return {"markings": [str(m.type) for m in data.crossed_lane_markings]}


# ------------------------------------------------------------------
# Sensor wrapper
# ------------------------------------------------------------------
def make_transform(d=None):
    d = d or {}
    return carla.Transform(
        carla.Location(x=d.get("x", 0.0), y=d.get("y", 0.0), z=d.get("z", 0.0)),
        carla.Rotation(pitch=d.get("pitch", 0.0), yaw=d.get("yaw", 0.0), roll=d.get("roll", 0.0)),
    )


class Sensor:
    """Spawns one CARLA sensor attached to a parent actor and buffers its output."""

    def __init__(self, world, parent, name, type, transform=None, attributes=None):
        if type not in SENSOR_REGISTRY:
            raise ValueError(f"Unknown sensor type '{type}'. Registered: {sorted(SENSOR_REGISTRY)}")
        self.name = name
        self.type = type
        self.spec = SENSOR_REGISTRY[type]

        bp = world.get_blueprint_library().find(self.spec.blueprint)
        for key, value in (attributes or {}).items():
            if not bp.has_attribute(key):
                raise ValueError(f"Sensor '{name}' ({self.spec.blueprint}) has no attribute '{key}'")
            bp.set_attribute(key, str(value))

        self.actor = world.spawn_actor(bp, make_transform(transform), attach_to=parent)
        self._queue = queue.Queue()
        self.actor.listen(self._queue.put)
        self._last = None

    def get(self, frame, sync=True, timeout=1.0):
        """Return this sensor's data for `frame`.

        Streaming sensors: the converted measurement (or the last one if none arrived).
        Event sensors: a list of events since the previous call (often empty).
        """
        if not self.spec.streaming:
            events = []
            while not self._queue.empty():
                events.append(self.spec.convert(self._queue.get_nowait()))
            return events

        if not sync:
            # Async mode: just take the newest measurement available.
            raw = None
            while not self._queue.empty():
                raw = self._queue.get_nowait()
            if raw is not None:
                self._last = self.spec.convert(raw)
            return self._last

        # Sync mode: wait for the measurement matching this frame, dropping older ones.
        try:
            while True:
                raw = self._queue.get(timeout=timeout)
                if raw.frame >= frame:
                    self._last = self.spec.convert(raw)
                    break
        except queue.Empty:
            pass  # sensor ticks slower than the world (sensor_tick > 0): reuse last value
        return self._last

    def clear(self):
        while not self._queue.empty():
            self._queue.get_nowait()

    def destroy(self):
        if self.actor is not None and self.actor.is_alive:
            self.actor.stop()
            self.actor.destroy()
        self.actor = None
