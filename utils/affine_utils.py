import cv2
import numpy as np


def get_affine_mat(center, src_size, dst_size, rot=0.0):
    """

    Args:
        center: ndarray of float or int, [2], (u, v)
        src_size: ndarray of float or int, [2], (w, h)
        dst_size: ndarray of float or int, [2], (w, h)
        rot: float, angle in rad

    Returns:
        affine_mat: ndarray of float, [2, 3]

    """
    def get_dir(src_point, rot):
        sn, cs = np.sin(rot), np.cos(rot)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result

    def get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    src_w = src_size[0]
    dst_w, dst_h = dst_size

    src_dir = get_dir([0, src_w * -0.5], rot)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:3, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:3, :] = get_3rd_point(dst[0, :], dst[1, :])

    affine_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return affine_mat


def affine_transform(pts_img, affine_mat):
    """

    Args:
        pts_img: ndarray of float, [N, 2], (u, v) points in pixel coordinates
        affine_mat: ndarray of float, [2, 3]

    Returns:
        pts_img: ndarray of float, [N, 2], (u, v) points in pixel coordinates

    """
    num_points = pts_img.shape[0]
    pts_img = np.concatenate([pts_img, np.ones((num_points, 1))], axis=1)
    pts_img = np.dot(affine_mat, pts_img.T).T

    return pts_img
