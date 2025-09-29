import torch
import numpy as np
import math
from typing import Optional
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from scene.cameras import Camera

# -- helper function for rotation --
def rotation_matrix_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """
    Build rotation matrix from yaw (around Y axis) and pitch (around X axis).
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    R_yaw = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy]
    ], dtype=np.float32)

    R_pitch = np.array([
        [1,  0,   0],
        [0, cp, -sp],
        [0, sp,  cp]
    ], dtype=np.float32)

    return R_pitch @ R_yaw


# -- new interactive camera --
class ViewerCamera(Camera):
    def __init__(
        self,
        FoVx: float,
        FoVy: float,
        W: int,
        H: int,
        data_device: str = "cuda",
    ):
        # Call base constructor with dummy params
        dummy_image = torch.zeros((3, H, W), dtype=torch.float32)
        super().__init__(
            colmap_id=-1,
            R=np.eye(3, dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            FoVx=FoVx,
            FoVy=FoVy,
            image=dummy_image,
            image_name="viewer",
            uid=-1,
            gt_alpha_mask=None,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
            data_device=data_device,
        )

        # interactive state
        self.position = np.array([0.0, 0.0, 3.0], dtype=np.float32)  # start 3 units away
        self.yaw = 0.0
        self.pitch = 0.0

        # override depth range
        self.znear = 0.01
        self.zfar = 100.0

        # projection (fixed unless FoV changes)
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar,
                                fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)
            .to(self.data_device)
        )

        # initial update
        self.update_matrices()

    # ---------- setters ----------
    def set_position(self, pos: np.ndarray):
        """Set absolute camera position in world space."""
        self.position = np.array(pos, dtype=np.float32)
        self.update_matrices()

    def set_rotation(self, yaw: float, pitch: float):
        """Set absolute yaw/pitch (in radians)."""
        self.yaw = float(yaw)
        self.pitch = np.clip(float(pitch), -math.pi/2 + 0.01, math.pi/2 - 0.01)
        self.update_matrices()

    def look_at(self, target: np.ndarray, distance: float = 3.0, up: np.ndarray = np.array([0,1,0])):
        """
        Reposition camera so it looks at 'target' from a 'distance'.
        """
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)

        # place camera on a circle (behind target by default)
        # e.g., start along +Z axis
        direction = np.array([0, 0, 1], dtype=np.float32)
        eye = target + direction * distance

        self.position = eye

        # compute forward vector
        forward = target - eye
        forward /= np.linalg.norm(forward)

        # yaw = atan2(x, z), pitch = asin(-y)
        self.yaw = math.atan2(forward[0], forward[2])
        self.pitch = math.asin(-forward[1])

        self.update_matrices()

    def look_at_orbit(self, target: np.ndarray, radius: float, azimuth: float, elevation: float):
        """
        Place camera on a sphere around target.
        azimuth, elevation in radians.
        """
        tx, ty, tz = target
        x = tx + radius * math.cos(elevation) * math.sin(azimuth)
        y = ty + radius * math.sin(elevation)
        z = tz + radius * math.cos(elevation) * math.cos(azimuth)

        self.position = np.array([x, y, z], dtype=np.float32)

        forward = target - self.position
        forward /= np.linalg.norm(forward)

        self.yaw = math.atan2(forward[0], forward[2])
        self.pitch = math.asin(-forward[1])

        self.update_matrices()


    def update_matrices(self):
        """Recompute transforms from position + orientation."""
        R = rotation_matrix_from_yaw_pitch(self.yaw, self.pitch)  # [3, 3]
        T = -R @ self.position

        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R
        view[:3, 3] = T
        self.world_view_transform = torch.tensor(view, device=self.data_device).T

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)

        self.camera_center = torch.tensor(self.position, device=self.data_device)

    # ---- movement controls ----
    def get_forward(self) -> np.ndarray:
        fx = math.sin(self.yaw) * math.cos(self.pitch)
        fy = -math.sin(self.pitch)
        fz = math.cos(self.yaw) * math.cos(self.pitch)
        return np.array([fx, fy, fz], dtype=np.float32)

    def get_right(self) -> np.ndarray:
        # right = cross(forward, world_up)
        forward = self.get_forward()
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        return right / np.linalg.norm(right)

    def get_up(self) -> np.ndarray:
        forward = self.get_forward()
        right = self.get_right()
        up = np.cross(right, forward)
        return up / np.linalg.norm(up)
    
    def move_forward(self, dist: float):
        # extract c2w (inverse of world_view_transform)
        c2w = self.world_view_transform.inverse().cpu().numpy().T
        forward = c2w[:3, 2]   # camera -Z is forward
        self.position += forward * dist
        self.update_matrices()

    def move_right(self, dist: float):
        c2w = self.world_view_transform.inverse().cpu().numpy().T
        right = c2w[:3, 0]     # camera +X is right
        self.position += right * dist
        self.update_matrices()

    def move_up(self, dist: float):
        c2w = self.world_view_transform.inverse().cpu().numpy().T
        up = c2w[:3, 1]        # camera +Y is up
        self.position += up * dist
        self.update_matrices()


    def orbit(self, dyaw: float, dpitch: float):
        self.yaw += dyaw
        self.pitch = np.clip(self.pitch + dpitch,
                             -math.pi/2 + 0.01, math.pi/2 - 0.01)
        self.update_matrices()
