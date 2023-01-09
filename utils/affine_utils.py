import cv2
import numpy as np


def get_affine_mat(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32)):
    """

    Args:
        center:
        scale:
        rot:
        output_size:
        shift:

    Returns:

    """
    def get_dir(src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result

    def get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    return trans, trans_inv


def affine_transform(pts_img, affine_mat):
    """

    Args:
        pts_img: ndarray of float32, [N, 2], (u, v) points in pixel coordinates
        affine_mat: ndarray of float32, [2, 3]

    Returns:
        pts_img: ndarray of float32, [N, 2], (u, v) points in pixel coordinates

    """
    num_points = pts_img.shape[0]
    pts_img = np.concatenate([pts_img, np.ones((num_points, 1))], axis=1)
    pts_img = np.dot(affine_mat, pts_img.T).T

    return pts_img