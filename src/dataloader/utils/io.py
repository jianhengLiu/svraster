import torch, torchvision

from wisp.ops.image.io import chw_to_hwc


def load_rgb(path, normalize=True):
    """Loads an image.

    Args:
        path (str): Path to the image.
        noramlize (bool): If True, will return [0,1] floating point values. Otherwise returns [0,255] ints.

    Returns:
        Image as an array of shape [H,W,C]
    """
    img = torchvision.io.read_image(path)
    if normalize:
        img = img.float() / 255.0
    return chw_to_hwc(img)


def load_depth(path, depth_scale_inv):
    """Load a depth image from path."""
    try:
        import cv2
        import numpy as np

        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if depth is not None:
            depth = depth.astype(np.float32) * depth_scale_inv
            return torch.from_numpy(depth)
    except Exception as e:
        print(f"Error loading depth {path}: {e}")
    return None
