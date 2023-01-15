import numpy as np

from utils.depth_map_utils import fill_in_fast


def get_points_in_fov(points, image, calib):
    """

    Args:
        points: ndarray of float32, [N, 3], points of (x, y, z)
        image: ndarray of uint8, [H, W, 3], RGB image
        calib: kitti_calibration_utils.Calibration

    Returns:
        points_in_fov: ndarray of float32, [N', 3], points of (x, y, z) in fov

    """
    h, w, _ = image.shape
    # pts_rect: [N, 3], points of (x, y, z) in camera coordinates
    pts_rect = calib.lidar_to_rect(points)
    # pts_img: [N, 2], points of (u, v) in image coordinates
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)

    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < w)
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < h)
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return points[pts_valid_flag]


def get_point_colors(points, image, calib):
    """

    Args:
        points: ndarray of float32, [N, 3], points of (x, y, z)
        image: ndarray of uint8, [H, W, 3], RGB image
        calib: kitti_calibration_utils.Calibration

    Returns:
        points_in_fov: ndarray of float32, [N', 3], points of (x, y, z) in fov
        point_colors: ndarray of float32, [N', 3], (r, g, b)

    """
    points_in_fov = get_points_in_fov(points, image, calib)
    pts_rect = calib.lidar_to_rect(points_in_fov)
    pts_img, _ = calib.rect_to_img(pts_rect)
    pts_img = pts_img.astype(np.int)
    rgb = image[pts_img[:, 1], pts_img[:, 0], :]
    return points_in_fov, rgb.astype(np.float32) / 255.0


def get_depth_map(points, image, calib):
    """

    Args:
        points: ndarray of float32, [N, 3], points of (x, y, z)
        image: ndarray of uint8, [H, W, 3], RGB image
        calib: kitti_calibration_utils.Calibration

    Returns:
        depth_map: ndarray of float32, [H, W], z values in camera coordinates

    """
    points_in_fov = get_points_in_fov(points, image, calib)
    h, w, _ = image.shape
    pts_rect = calib.lidar_to_rect(points_in_fov)
    pts_img, _ = calib.rect_to_img(pts_rect)
    pts_img = pts_img.astype(np.int)
    depth_map = np.zeros((h, w), dtype=np.float32)
    depth_map[pts_img[:, 1], pts_img[:, 0]] = pts_rect[:, -1]
    return depth_map


def get_completed_depth_map(points, image, calib):
    """

    Args:
        points: ndarray of float32, [N, 3], points of (x, y, z)
        image: ndarray of uint8, [H, W, 3], RGB image
        calib: kitti_calibration_utils.Calibration

    Returns:
        depth_map: ndarray of float32, [H, W], z values in camera coordinates

    """
    depth_map = get_depth_map(points, image, calib)
    depth_map = fill_in_fast(depth_map, extrapolate=False, blur_type='bilateral')
    return depth_map


def compute_projection_map_from_depth_map(depth_map, calib):
    """

    Args:
        depth_map: ndarray of float32, [H, W], z values in camera coordinates
        calib: kitti_calibration_utils.Calibration

    Returns:
        projection_map: ndarray of float32, [H, W, 3], (x, y, z) values in camera coordinates

    """
    h, w = depth_map.shape
    u = np.arange(w)[None, :].repeat(h, axis=0).astype(np.float32)
    x_camera = (u - calib.cu) * depth_map / calib.fu
    v = np.arange(h)[:, None].repeat(w, axis=1).astype(np.float32)
    y_camera = (v - calib.cv) * depth_map / calib.fv
    return np.stack([x_camera, y_camera, depth_map], axis=-1)


def get_lidar_projection_map(points, image, calib):
    """

    Args:
        points: ndarray of float32, [N, 3], points of (x, y, z)
        image: ndarray of uint8, [H, W, 3], RGB image
        calib: kitti_calibration_utils.Calibration

    Returns:
        projection_map: ndarray of float32, [H, W, 3], (x, y, z) values in camera coordinates

    """
    depth_map = get_depth_map(points, image, calib)
    return compute_projection_map_from_depth_map(depth_map, calib)


def get_completed_lidar_projection_map(points, image, calib):
    """

    Args:
        points: ndarray of float32, [N, 3], points of (x, y, z)
        image: ndarray of uint8, [H, W, 3], RGB image
        calib: kitti_calibration_utils.Calibration

    Returns:
        projection_map: ndarray of float32, [H, W, 3], (x, y, z) values in camera coordinates

    """
    depth_map = get_completed_depth_map(points, image, calib)
    return compute_projection_map_from_depth_map(depth_map, calib)

