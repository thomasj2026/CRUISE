"""Simple pygame viewer: shows one sensor plus a text HUD. Close the window or press ESC to quit."""
import numpy as np
import pygame

_PALETTE = np.random.default_rng(0).integers(0, 255, size=(256, 3), dtype=np.uint8)
_PALETTE[0] = 0


def sensor_to_image(data, sensor_type, lidar_range=50.0, bev_size=600):
    """Turn sensor output into an (H, W, 3) uint8 image, or None if it can't be shown."""
    if data is None:
        return None
    if sensor_type == "rgb_camera":
        return data
    if sensor_type == "depth_camera":
        d = np.clip(np.log1p(data) / np.log1p(1000.0), 0.0, 1.0)
        gray = (255 * (1.0 - d)).astype(np.uint8)
        return np.stack([gray] * 3, axis=-1)
    if sensor_type == "semantic_camera":
        return _PALETTE[data]
    if sensor_type == "lidar":
        img = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        scale = (bev_size / 2) / lidar_range
        cols = (bev_size / 2 + data[:, 1] * scale).astype(int)   # y (right) -> image x
        rows = (bev_size / 2 - data[:, 0] * scale).astype(int)   # x (forward) -> image up
        ok = (rows >= 0) & (rows < bev_size) & (cols >= 0) & (cols < bev_size)
        img[rows[ok], cols[ok]] = (255, 255, 255)
        c = bev_size // 2
        img[c - 4:c + 4, c - 2:c + 2] = (255, 60, 60)   # ego car
        return img
    return None


class Display:
    def __init__(self, sensor_name, sensor_type, lidar_range=50.0):
        pygame.init()
        pygame.display.set_caption("CARLA AV framework")
        self.sensor_name = sensor_name
        self.sensor_type = sensor_type
        self.lidar_range = lidar_range
        self.font = pygame.font.SysFont("monospace", 16)
        self.screen = None

    def render(self, obs, control, info=None, stats=None):
        """Draw a frame. Returns False when the user wants to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        img = sensor_to_image(obs["sensors"].get(self.sensor_name), self.sensor_type, self.lidar_range)
        if img is None:
            img = np.zeros((600, 800, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        if self.screen is None or self.screen.get_size() != (w, h):
            self.screen = pygame.display.set_mode((w, h))
        self.screen.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))

        lines = [
            f"frame {obs['frame']}   t={obs['timestamp']:.1f}s",
            f"speed {obs['ego']['speed'] * 3.6:5.1f} km/h",
        ]
        if control is not None:
            lines.append(f"thr {control.throttle:.2f}  steer {control.steer:+.2f}  brake {control.brake:.2f}")
        for k, v in {**(stats or {}), **(info or {})}.items():
            lines.append(f"{k}: {v}")
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (255, 255, 255), (0, 0, 0))
            self.screen.blit(surf, (8, 8 + i * 20))

        pygame.display.flip()
        return True

    def close(self):
        pygame.quit()
