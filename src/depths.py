import torch
from typing import Optional

NORMAL_COMPUTATION_AVAILABLE = True


class Depth:
    def __init__(self, w2c, depth_name, origin, dirs, depths):
        self.w2c = w2c
        self.depth_name = depth_name
        self.origin = origin
        self.dirs = dirs
        self.depths = depths


class Depths:
    origins: torch.Tensor  # (B, 1, 3)
    depths: torch.Tensor  # (B, N, 1)
    dirs: torch.Tensor  # (B, N, 3) in world coordinates
    normals: torch.Tensor  # (B, N, 3) in world coordinates
    name: str = None  # Optional name for the depth map

    def __init__(
        self,
        origins,
        depths,
        dirs,
        normals=None,
        name=None,
    ):
        self.origins = origins
        self.depths = depths
        self.dirs = dirs
        self.normals = normals
        self.name = name

        # check shapes
        if (
            self.origins.ndim != 3
            or self.origins.shape[1] != 1
            or self.origins.shape[2] != 3
        ):
            raise ValueError("Origins must have shape (B, 1, 3).")
        if (
            self.depths.ndim != 3
            or self.depths.shape[1] != self.dirs.shape[1]
            or self.depths.shape[2] != 1
        ):
            raise ValueError("Depths must have shape (B, N, 1) where N matches dirs.")
        if (
            self.dirs.ndim != 3
            or self.dirs.shape[1] != self.depths.shape[1]
            or self.dirs.shape[2] != 3
        ):
            raise ValueError("Dirs must have shape (B, N, 3) where N matches depths.")
        print(
            f"Depths initialized with origins shape: {self.origins.shape}, "
            f"depths shape: {self.depths.shape}, dirs shape: {self.dirs.shape}"
        )

        self.depth_static = torch.tensor(
            (self.depths.shape[0], self.depths.shape[1]),
        )
        self.estimate_map_size()

    def to(self, device):
        self.origins = self.origins.to(device)
        self.depths = self.depths.to(device)
        self.dirs = self.dirs.to(device)
        if self.normals is not None:
            self.normals = self.normals.to(device)
        return self

    def get_xyzs(self, sample_num: int = None):
        """Convert depths to xyzs."""
        if sample_num is not None:
            sample_idx = (torch.rand((sample_num, 2)) * self.depth_static).to(
                torch.long
            )
            c, y = (i.flatten() for i in torch.split(sample_idx, 1, dim=-1))
            return self.origins[c, 0] + self.depths[c, y] * self.dirs[c, y]
        return self.origins + self.depths * self.dirs

    def estimate_map_size(self):
        batch_num = 10000
        total_pt_num = self.depths.shape[0] * self.depths.shape[1]
        if total_pt_num > batch_num:
            xyzs = self.get_xyzs(batch_num).view(-1, 3)  # Flatten to (B*N, 3)
        else:
            xyzs = self.get_xyzs().view(-1, 3)  # Flatten to (B*N, 3)
        self.xyzs_center, _ = xyzs.median(dim=0, keepdim=True)
        self.xyzs_radius = (xyzs - self.xyzs_center).norm(dim=1).max().item()
        self.inner_map_size = self.xyzs_radius * 2.0

        print(
            f"Estimated map size: {self.inner_map_size:.3f} (center: {self.xyzs_center}, radius: {self.xyzs_radius:.3f})"
        )
        # self.octree_level = (
        #     np.ceil(
        #         np.log2((self.inner_map_size + 2 * self.leaf_size) / self.leaf_size)
        #     )
        #     .astype(np.int8)
        #     .item()
        # )
        # print(f"leaf_size: {self.leaf_size}, octree_level: {self.octree_level}")
