# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch

from src.depths import Depth, Depths
from src.dataloader.utils.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from src.utils.camera_utils import focal2fov
import plyfile


def read_custom_colmap_dataset(
    source_path, image_dir_name, test_every, use_test, camera_creator
):

    source_path = Path(source_path)

    # Parse colmap meta data
    sparse_path = source_path / "sparse" / "0"
    if not sparse_path.exists():
        sparse_path = source_path / "colmap" / "sparse" / "0"
    if not sparse_path.exists():
        raise Exception("Can not find COLMAP reconstruction.")

    try:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

        try:
            depths_extrinsic_file = os.path.join(sparse_path, "depths.bin")
            depth_extrinsics = read_extrinsics_text(depths_extrinsic_file)
            print("Depth extrinsics loaded from binary file.")
        except:
            print("Depth extrinsics not found in binary format.")
            depth_extrinsics = None
    except:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        try:
            depths_extrinsic_file = os.path.join(sparse_path, "depths.txt")
            depth_extrinsics = read_extrinsics_text(depths_extrinsic_file)
        except:
            print("Depth extrinsics not found in text format.")
            depth_extrinsics = None

    cam_lst, point_cloud = read_colmap_cameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(source_path, image_dir_name),
        camera_creator=camera_creator,
    )
    cam_lst = sorted(cam_lst, key=lambda x: x.image_name)

    # Split train/test
    if use_test:
        train_cam_lst = [cam for i, cam in enumerate(cam_lst) if i % test_every != 0]
        test_cam_lst = [cam for i, cam in enumerate(cam_lst) if i % test_every == 0]
    else:
        train_cam_lst = cam_lst
        test_cam_lst = []

    if depth_extrinsics is not None:
        # TODO: add depth_dir_name
        depth_lst = read_colmap_depths(
            depth_extrinsics=depth_extrinsics,
            depths_folder=os.path.join(source_path, "depths"),
            camera_creator=camera_creator,
        )
        depth_lst = sorted(depth_lst, key=lambda x: x.depth_name)

        if use_test:
            train_depth_lst = [
                depth for i, depth in enumerate(depth_lst) if i % test_every != 0
            ]
            test_depth_lst = [
                depth for i, depth in enumerate(depth_lst) if i % test_every == 0
            ]
        else:
            train_depth_lst = depth_lst
            test_depth_lst = []

    # suggested_center = train_dataset.xyzs_center.squeeze(0).numpy()
    # suggested_radius = train_dataset.xyzs_radius
    # suggested_bounding = np.stack(
    #     [
    #         suggested_center - suggested_radius,
    #         suggested_center + suggested_radius,
    #     ]
    # )
    suggested_bounding = None

    point_cloud = None
    # Pack dataset
    dataset = {
        "train_cam_lst": train_cam_lst,
        "test_cam_lst": test_cam_lst,
        "suggested_bounding": suggested_bounding,
        "point_cloud": point_cloud,
    }
    if depth_extrinsics is not None:
        dataset["train_depth_lst"] = train_depth_lst
        dataset["test_depth_lst"] = test_depth_lst

        # Convert train_depth_lst to train_depths (similar to replica dataset)
        if train_depth_lst:
            # Stack origins, depths, and dirs from all depth objects
            origins_list = []
            depths_list = []
            dirs_list = []

            for depth_obj in tqdm(train_depth_lst, desc="Processing depth maps"):
                # Add batch dimension to origin and expand to (1, 1, 3)
                origins_list.append(depth_obj.origin.unsqueeze(0).unsqueeze(0))

                # Add batch dimension to depths (N, 1) -> (1, N, 1)
                depths_list.append(depth_obj.depths.unsqueeze(0))

                # Add batch dimension to dirs (N, 3) -> (1, N, 3)
                dirs_list.append(depth_obj.dirs.unsqueeze(0))

            # Concatenate all depth data along batch dimension
            train_depths = Depths(
                origins=torch.cat(origins_list, dim=0),  # (B, 1, 3)
                depths=torch.cat(depths_list, dim=0),  # (B, N, 1)
                dirs=torch.cat(dirs_list, dim=0),  # (B, N, 3)
                normals=None,  # Will be computed later if needed
            )
            dataset["train_depths"] = train_depths

            # Update suggested_bounding using train_depths
            suggested_center = train_depths.xyzs_center.squeeze(0).numpy()
            suggested_radius = train_depths.xyzs_radius
            suggested_bounding = np.stack(
                [
                    suggested_center - suggested_radius,
                    suggested_center + suggested_radius,
                ]
            )
            dataset["suggested_bounding"] = suggested_bounding

    return dataset


def read_colmap_cameras(cam_extrinsics, cam_intrinsics, images_folder, camera_creator):
    cam_lst = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        image_path = Path(os.path.join(images_folder, extr.name))
        if not image_path.exists():
            image_path = image_path.with_suffix(".png")
        if not image_path.exists():
            image_path = image_path.with_suffix(".jpg")
        if not image_path.exists():
            image_path = image_path.with_suffix(".JPG")
        if not image_path.exists():
            raise Exception(f"File not found: {str(image_path)}")
        image = Image.open(image_path)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            fovx = focal2fov(focal_length_x, width)
            fovy = focal2fov(focal_length_x, height)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            fovx = focal2fov(focal_length_x, width)
            fovy = focal2fov(focal_length_y, height)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = T

        cam_info = camera_creator(
            image=image,
            w2c=w2c,
            fovx=fovx,
            fovy=fovy,
            cx_p=intr.params[2] / width,
            cy_p=intr.params[3] / height,
            sparse_pt=None,
            image_name=image_path.name,
        )
        cam_lst.append(cam_info)

    point_cloud = None
    sys.stdout.write("\n")
    return cam_lst, point_cloud


def read_colmap_depths(depth_extrinsics, depths_folder, camera_creator):
    depth_lst = []
    for idx, key in enumerate(depth_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading depth {}/{}".format(idx + 1, len(depth_extrinsics)))
        sys.stdout.flush()

        extr = depth_extrinsics[key]

        depth_path = Path(os.path.join(depths_folder, extr.name))
        if not depth_path.exists():
            raise Exception(f"File not found: {str(depth_path)}")

        depth_image = None
        pointcloud = None
        if depth_path.suffix.lower() == ".png":
            depth_image = torch.tensor(
                np.array(Image.open(depth_path)).astype(np.float32)
            )
        elif depth_path.suffix.lower() == ".ply" or depth_path.suffix.lower() == ".pcd":
            plydata = plyfile.PlyData.read(str(depth_path))
            vertices = plydata["vertex"]
            pointcloud = torch.from_numpy(
                np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            )
        else:
            raise Exception(f"Unsupported depth file format: {str(depth_path)}")

        R = torch.tensor(qvec2rotmat(extr.qvec), dtype=torch.float32)
        T = torch.tensor(extr.tvec, dtype=torch.float32)
        w2l = torch.eye(4, dtype=torch.float32)
        w2l[:3, :3] = R
        w2l[:3, 3] = T

        origin = -R @ T

        if depth_image is not None:
            raise NotImplementedError("Depth image handling not implemented")
        elif pointcloud is not None:
            dists = pointcloud.norm(dim=-1, keepdim=True)
            depths = dists
            dirs_L = pointcloud / dists
            dirs = (R.T @ dirs_L.T).T

        # unify depth number into ds_pt_num=1e5
        ds_pt_num = 1e4
        if depths.numel() > ds_pt_num:
            sample_idx = torch.randperm(depths.shape[0])[: int(ds_pt_num)]
            depths = depths[sample_idx]
            dirs = dirs[sample_idx]

        depth_info = Depth(
            w2c=w2l, depth_name=depth_path.name, origin=origin, dirs=dirs, depths=depths
        )
        depth_lst.append(depth_info)

    sys.stdout.write("\n")
    return depth_lst
