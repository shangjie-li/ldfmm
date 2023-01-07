import numpy as np


def rotate_points_along_z(points, angle):
    """

    Args:
        points: ndarray of float32, [N, 3], (x, y, z) in lidar coordinates
        angle: float, angle along z-axis, angle increases x ==> y

    Returns:
        points: ndarray of float32, [N, 3], (x, y, z) in lidar coordinates

    """
    rot_matrix = np.array([
        np.cos(angle), -np.sin(angle), 0,
        np.sin(angle), np.cos(angle), 0,
        0, 0, 1,
    ]).reshape(3, 3)
    points = np.dot(rot_matrix, points.T).T
    return points


def box3d_lidar_to_corners3d(box3d_lidar):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        box3d_lidar: ndarray of float32, [7], (x, y, z, l, w, h, heading) in lidar coordinates

    Returns:
        corners3d: ndarray of float32, [8, 3], (x, y, z) in lidar coordinates

    """
    box3d_lidar = box3d_lidar.reshape(1, 7)
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ]) / 2
    corners3d = box3d_lidar[:, 3:6].repeat(8, axis=0) * template
    corners3d = rotate_points_along_z(corners3d.reshape(8, 3), box3d_lidar[0, 6]).reshape(8, 3)
    corners3d += box3d_lidar[:, 0:3]
    return corners3d


def boxes3d_camera_to_lidar(boxes3d, calib):
    """

    Args:
        boxes3d: ndarray of float32, [N, 7], (x, y, z, h, w, l, ry] in camera coordinates
        calib: kitti_calibration_utils.Calibration

    Returns:
        boxes3d: ndarray of float32, [N, 7], (x, y, z, l, w, h, heading) in lidar coordinates

    """
    boxes3d = boxes3d.copy()
    xyz, ry = boxes3d[:, 0:3], boxes3d[:, 6:7]
    h, w, l = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6]
    xyz = calib.rect_to_lidar(xyz)
    heading = -np.pi / 2 - ry
    return np.concatenate([xyz, l, w, h, heading], axis=-1)