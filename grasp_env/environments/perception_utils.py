from typing import List, Optional

from grasp_env.simulator.perception import create_tsdf
from grasp_env.simulator.transform import Transform


import numpy as np
import open3d as o3d

def _build_frames(depth_imgs, intrinsic, extrinsics):
    """Pack the per-frame data we need for visibility checks."""
    K = np.array([[intrinsic.fx, 0, intrinsic.cx],
                  [0, intrinsic.fy, intrinsic.cy],
                  [0, 0, 1]], dtype=float)
    W, H = intrinsic.width, intrinsic.height
    frames = []
    for D, ext in zip(depth_imgs, extrinsics):
        ext = Transform.from_list(ext)
        frames.append({
            "K": K,
            "size": (W, H),
            "Twc": ext.inverse().as_matrix(),  # cam->world
            "Tcw": ext.as_matrix(),            # world->cam
            "depth": D.astype(np.float64),
        })
    return frames

def orient_normals_visible_simple(
    pcd: o3d.geometry.PointCloud,
    frames: List[dict],
    depth_tol_abs: float = 0.02,   # 2 cm absolute tolerance
    depth_tol_rel: float = 0.02,   # 2% of depth (helps at distance)
):
    """Fast, compact multi-view normal orientation with depth visibility checks."""
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
        )

    pts  = np.asarray(pcd.points)
    nors = np.asarray(pcd.normals)
    N    = pts.shape[0]
    eps  = 1e-12

    v_sum = np.zeros((N, 3), dtype=np.float64)
    w_sum = np.zeros((N,),    dtype=np.float64)
    cams  = np.stack([fr["Twc"][:3, 3] for fr in frames], axis=0)  # (M,3)

    for fr in frames:
        R = fr["Tcw"][:3, :3]; t = fr["Tcw"][:3, 3]
        Pc = pts @ R.T + t               # (N,3) world->cam
        Z  = Pc[:, 2]
        zpos = Z > 0
        if not np.any(zpos):
            continue

        fx, fy, cx, cy = fr["K"][0,0], fr["K"][1,1], fr["K"][0,2], fr["K"][1,2]
        u = (Pc[zpos, 0] * fx) / Z[zpos] + cx
        v = (Pc[zpos, 1] * fy) / Z[zpos] + cy
        W, H = fr["size"]
        inb = (u >= 0) & (v >= 0) & (u < W) & (v < H)
        if not np.any(inb):
            continue

        u  = u[inb];  v = v[inb]
        idx = np.nonzero(zpos)[0][inb]

        ui = np.floor(u).astype(np.int32)
        vi = np.floor(v).astype(np.int32)
        d_img = fr["depth"][vi, ui]
        z = Z[idx]
        tol = np.maximum(depth_tol_abs, depth_tol_rel * z)
        vis = (d_img > 0) & (np.abs(d_img - z) <= tol)
        if not np.any(vis):
            continue

        idx = idx[vis]
        c = fr["Twc"][:3, 3]          # camera center
        vdir = c - pts[idx]           # point->camera keeps outward vote
        d2 = np.einsum('ij,ij->i', vdir, vdir).clip(min=eps)
        vhat = vdir / np.sqrt(d2)[:, None]
        w = 1.0 / d2

        np.add.at(v_sum, idx, vhat * w[:, None])
        np.add.at(w_sum, idx, w)

    # Fallback: if a point had no visible votes, use all cameras (IDW)
    need_fb = (w_sum == 0)
    if np.any(need_fb):
        diff = cams[None, :, :] - pts[need_fb][:, None, :]  # (K,M,3)
        d2   = np.einsum('kmi,kmi->km', diff, diff).clip(min=eps)
        vhat = diff / np.sqrt(d2)[:, :, None]
        w    = (1.0 / d2)
        w   /= w.sum(axis=1, keepdims=True)
        v_avg_fb = (w[:, :, None] * vhat).sum(axis=1)
        v_sum[need_fb] = v_avg_fb
        w_sum[need_fb] = 1.0

    v_avg = v_sum / w_sum[:, None]
    flip = (np.einsum('ij,ij->i', nors, v_avg) < 0)
    nors[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(nors)
    return pcd


def cloud_from_image_data(
    image_data: dict,
    voxelization_size: float = 0.0045,
    lower_bounds: np.ndarray = np.array([0, 0, 0]),
    upper_bounds: np.ndarray = np.array([0, 0, 1]),
    tsdf_size: float = 0.3,
    tsdf_resolution: int = 180,
    noise_level: float = 0.0008,
) -> Optional[o3d.geometry.PointCloud]:
    depth_imgs = image_data["depth_imgs"]
    rgb_imgs = image_data["rgb_imgs"]
    intrinsic = image_data["intrinsics"]
    extrinsics = image_data["extrinsics"]
    eye = image_data["eye"]

    # Fuse image into tsdf
    tsdf = create_tsdf(tsdf_size, tsdf_resolution, depth_imgs, intrinsic, extrinsics, rgb_imgs=rgb_imgs)
    pcd = tsdf.get_cloud()

    bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower_bounds, upper_bounds)

    pcd = pcd.crop(bounding_box)
    if len(pcd.points) < 50:
        return None

    # Optional jitter
    if noise_level > 0:
        vertices = np.asarray(pcd.points)
        vertices = vertices + np.random.normal(loc=0.0, scale=noise_level, size=(len(vertices), 3))
        pcd.points = o3d.utility.Vector3dVector(vertices)

    # remove outliers & smooth normals
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    pcd, _ = pcd.remove_radius_outlier(12, 3 * voxelization_size)

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=3.5 * voxelization_size, max_nn=600)
    )
    pcd.normalize_normals()

    # optional for ICP only:
    # this is redundant since I am using orient_normals_visible_simple
    # pcd.orient_normals_consistent_tangent_plane(50)


    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    # pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
    # pcd.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
    # )
    # pcd.orient_normals_consistent_tangent_plane(30)

    # >>> Multi-view orientation (visibility-aware)
    frames = _build_frames(depth_imgs, intrinsic, extrinsics)
    pcd = orient_normals_visible_simple(pcd, frames)

    # downsample
    pcd = pcd.voxel_down_sample(voxel_size=voxelization_size)
    return pcd


def refine_image_data(
    image_data: dict,
    voxel_size=0.002,
    depth_trunc=0.7,
    max_correspondence_distance=0.015,
):
    """
    Args:
        image_data: dict with keys "rgb_imgs", "depth_imgs", "intrinsics" (CameraIntrinsic),
            and "extrinsics" (T_world2cam as list-like of length 7)
        voxel_size: downsampling size for ICP (meters)
        depth_trunc: max depth (meters) used for RGBD creation

    Returns:
        image_data dict with the "extrinsics" refined (still T_world2cam)
    """
    depth_images = image_data["depth_imgs"]
    rgb_images = image_data["rgb_imgs"]
    intrinsic = image_data["intrinsics"]
    extrinsics_w2c = np.asarray(image_data["extrinsics"])

    if len(depth_images) == 0:
        return image_data

    # 1) Project each RGB-D into world-space point cloud and crop
    pcds = []
    intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsic.width),
        int(intrinsic.height),
        float(intrinsic.fx),
        float(intrinsic.fy),
        float(intrinsic.cx),
        float(intrinsic.cy),
    )
    T_cam2world_list = []
    for rgb, depth, T_world2cam in zip(rgb_images, depth_images, extrinsics_w2c):
        color = np.ascontiguousarray(rgb[..., ::-1])  # BGR -> RGB without extra dependency
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth))
        color_o3d = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
        T_cam2world = Transform.from_list(T_world2cam).inverse().as_matrix()
        pcd.transform(T_cam2world)  # camera -> world
        pcd = pcd.voxel_down_sample(voxel_size)
        if len(pcd.points) > 0:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.0)
        # pcd = pcd.crop(bbox)
        pcds.append(pcd)
        T_cam2world_list.append(T_cam2world)

    if len(pcds) == 0:
        return image_data

    # 2) Fix first point cloud as reference
    target = pcds[0]
    target_down = target.voxel_down_sample(voxel_size)
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0, max_nn=30
        )
    )

    refined_T_cam2world = [np.array(T) for T in T_cam2world_list]

    # choose estimation method
    estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=30,
    )

    # 3) Register all other clouds to the first
    for i in range(1, len(pcds)):
        source = pcds[i]

        source_down = source.voxel_down_sample(voxel_size)
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.0, max_nn=30
            )
        )

        reg = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            max_correspondence_distance=max_correspondence_distance,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=criteria,
        )

        # Apply refinement to the *original* full-res cloud and its pose
        pcds[i].transform(reg.transformation)
        refined_T_cam2world[i] = reg.transformation @ refined_T_cam2world[i]

    refined_extrinsics = np.array(
        [Transform.from_matrix(T).inverse().to_list() for T in refined_T_cam2world],
        dtype=extrinsics_w2c.dtype if isinstance(extrinsics_w2c, np.ndarray) else np.float64,
    )
    refined_eyes = [Transform.from_matrix(T).translation for T in refined_T_cam2world]

    refined_image_data = dict(image_data)
    refined_image_data["extrinsics"] = refined_extrinsics
    refined_image_data["eye"] = refined_eyes
    return refined_image_data

