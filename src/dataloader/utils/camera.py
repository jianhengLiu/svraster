from kaolin.render.camera import Camera
from kaolin.render.camera.intrinsics_pinhole import PinholeIntrinsics
from wisp.core import Rays
import torch


class OpencvPinholeIntrinsics(PinholeIntrinsics):
    """A pinhole camera intrinsics class that uses OpenCV conventions for the principal point and focal length.

    This class is a wrapper around the Kaolin PinholeIntrinsics class, which allows for the use of OpenCV conventions
    for the principal point and focal length. The principal point is specified in pixel coordinates,
    and the focal length is specified in pixels.
    """

    DEFAULT_NEAR = 1e-2
    DEFAULT_FAR = 1e2

    def __init__(
        self,
        width: int,
        height: int,
        params: torch.Tensor,
        near: float = DEFAULT_NEAR,
        far: float = DEFAULT_FAR,
    ):
        super().__init__(
            width=width,
            height=height,
            params=params,
            near=near,
            far=far,
        )

    # https://zhuanlan.zhihu.com/p/635801612
    def projection_matrix(self) -> torch.Tensor:
        l = -self.cx * self.near / self.focal_x
        r = (self.width - self.cx) * self.near / self.focal_x
        t = (self.height - self.cy) * self.near / self.focal_y
        b = -self.cy * self.near / self.focal_y

        project_matrix = torch.tensor(
            [
                [2 * self.near / (r - l), 0, -(r + l) / (r - l), 0],
                [0, 2 * self.near / (t - b), -(t + b) / (t - b), 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.near - self.far),
                    2 * (self.far * self.near) / (self.near - self.far),
                ],
                [0, 0, 1, 0],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        return project_matrix.unsqueeze(0)


def get_image_coords_zdir(camera: Camera, coords_grid: torch.Tensor):
    """Default ray generation function for pinhole cameras.

    This function assumes that the principal point (the pinhole location) is specified by a
    displacement (camera.x0, camera.y0) in pixel coordinates from the center of the image.

    The Kaolin camera class does not enforce a coordinate space for how the principal point is specified,
    so users will need to make sure that the correct principal point conventions are followed for
    the cameras passed into this function.

    Args:
        camera (kaolin.render.camera): The camera class.
        coords_grid (torch.FloatTensor): Grid of coordinates of shape [H, W, 2].

    Returns:
        (wisp.core.Rays): The generated pinhole rays for the camera.
    """
    if camera.device != coords_grid[0].device:
        raise Exception(
            f"Expected camera and coords_grid[0] to be on the same device, but found {camera.device} and {coords_grid[0].device}."
        )
    if camera.device != coords_grid[1].device:
        raise Exception(
            f"Expected camera and coords_grid[1] to be on the same device, but found {camera.device} and {coords_grid[1].device}."
        )
    # coords_grid should remain immutable (a new tensor is implicitly created here)
    pixel_y, pixel_x = coords_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)

    # Account for principal point (offsets from the center)
    pt_x = (pixel_x - camera.cx) / camera.focal_x
    pt_y = (pixel_y - camera.cy) / camera.focal_y

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x

    ray_dir = torch.stack(
        (
            pt_x,
            pt_y,
            torch.ones_like(pt_x),
        ),
        dim=-1,
    )

    ray_dir = ray_dir.reshape(-1, 3)  # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_orig, ray_dir = ray_orig[0], ray_dir[0]  # Assume a single camera

    return Rays(
        origins=ray_orig, dirs=ray_dir, dist_min=camera.near, dist_max=camera.far
    )


def get_image_coords_ndir(camera: Camera, coords_grid: torch.Tensor):
    """Default ray generation function for pinhole cameras.

    This function assumes that the principal point (the pinhole location) is specified by a
    displacement (camera.x0, camera.y0) in pixel coordinates from the center of the image.

    The Kaolin camera class does not enforce a coordinate space for how the principal point is specified,
    so users will need to make sure that the correct principal point conventions are followed for
    the cameras passed into this function.

    Args:
        camera (kaolin.render.camera): The camera class.
        coords_grid (torch.FloatTensor): Grid of coordinates of shape [H, W, 2].

    Returns:
        (wisp.core.Rays): The generated pinhole rays for the camera.
    """

    ray = get_image_coords_zdir(camera, coords_grid)
    z_norm = torch.linalg.norm(ray.dirs, dim=-1, keepdim=True)
    ray.dirs /= z_norm

    return dict(ray=ray, z_norm=z_norm)


def generate_rays_from_coords(x, y, fx, fy, cx, cy):
    # [height width 3]
    zrays = torch.stack(
        [
            (x - cx) / (fx),
            (y - cy) / (fy),
            torch.ones_like(x),
        ],
        dim=-1,
    )

    rays_norm = torch.norm(zrays, p=2, dim=-1, keepdim=True)
    nrays = zrays / rays_norm

    return [nrays, rays_norm]

# from kaolin-wisp
def generate_default_grid(width, height, device=None):
    h_coords = torch.arange(height, device=device, dtype=torch.float)
    w_coords = torch.arange(width, device=device, dtype=torch.float)
    return torch.meshgrid(h_coords, w_coords)  # return pixel_y, pixel_x

def generate_centered_pixel_coords(img_width, img_height, res_x=None, res_y=None, device=None):
    pixel_y, pixel_x = generate_default_grid(res_x, res_y, device)
    scale_x = 1.0 if res_x is None else float(img_width) / res_x
    scale_y = 1.0 if res_y is None else float(img_height) / res_y
    pixel_x = pixel_x * scale_x + 0.5   # scale and add bias to pixel center
    pixel_y = pixel_y * scale_y + 0.5   # scale and add bias to pixel center
    return pixel_y, pixel_x