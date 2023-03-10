import numpy as np
import torch

from utils.box_utils import boxes3d_camera_to_lidar
from utils.box_utils import box3d_lidar_to_corners3d
from ops.iou3d_nms.iou3d_nms_utils import nms_gpu


def normalize_angle(angle):
    """

    Args:
        angle: float or ndarray of float, angle in rad

    Returns:
        angle: float or ndarray of float, angle in rad

    """
    sina = np.sin(angle)
    cosa = np.cos(angle)

    return np.arctan2(sina, cosa)


def bin_to_angle(bin_id, residual_angle):
    """

    Args:
        bin_id: int, bin id (0 to 11)
        residual_angle: float, angle in rad

    Returns:
        angle: float, angle in rad

    """
    angle_per_bin = 2 * np.pi / 12
    angle_center = bin_id * angle_per_bin
    angle = angle_center + residual_angle

    return normalize_angle(angle)


def nms(objects, calib, nms_thresh):
    """

    Args:
        objects: list
        calib: kitti_calibration_utils.Calibration
        nms_thresh: float

    Returns:
        selected: list

    """
    objects = np.array(objects).astype(np.float32).reshape(-1, 14)
    scores = objects[:, -1]
    sizes3d, center3d, rys = objects[:, 6:9], objects[:, 9:12], objects[:, 12:13]
    center3d[:, 1] += -sizes3d[:, 0] / 2
    boxes3d = np.concatenate([center3d, sizes3d, rys], axis=-1)
    boxes3d_lidar = boxes3d_camera_to_lidar(boxes3d, calib)

    selected = []
    if scores.shape[0] > 0:
        selected, _ = nms_gpu(torch.from_numpy(boxes3d_lidar).cuda(), torch.from_numpy(scores).cuda(), nms_thresh)
        selected = selected.tolist()

    return selected


def decode_detections(preds, infos, calibs, regress_box2d, score_thresh, nms_thresh):
    """

    Args:
        preds: dict
        infos: dict
        calibs: list
        regress_box2d: bool
        score_thresh: float
        nms_thresh: float

    Returns:
        det: dict

    """
    det = {}
    batch_size, K, _ = preds['cls_id'].shape
    for i in range(batch_size):
        img_id = infos['img_id'][i]
        img_size = infos['img_size'][i]
        downsample = infos['original_downsample'][i]
        calib = calibs[i]
        det_per_img = []
        for j in range(K):
            cls_id = preds['cls_id'][i, j, 0]
            score = preds['score'][i, j, 0]
            center3d_img = preds['center3d_img'][i, j, :]
            depth_ref = preds['depth_ref'][i, j, 0]
            depth_res = preds['depth_res'][i, j, 0]
            size3d = preds['size3d'][i, j, :]
            alpha_bin = preds['alpha_bin'][i, j, :]
            alpha_res = preds['alpha_res'][i, j, :]

            if score < score_thresh: continue
            if depth_ref == 0.0: continue

            depth = depth_ref + depth_res

            x_img, y_img = center3d_img * downsample
            center3d = calib.img_to_rect(x_img, y_img, depth).reshape(-1)
            loc = center3d + [0, size3d[0] / 2, 0]

            bin_id = np.argmax(alpha_bin)
            residual_angle = alpha_res[bin_id]
            alpha = bin_to_angle(bin_id, residual_angle)
            ry = alpha + np.arctan2(center3d[0], center3d[2])

            if regress_box2d:
                center2d = preds['center2d'][i, j, :] * downsample
                size2d = preds['size2d'][i, j, :] * downsample
                u1, v1 = center2d - size2d / 2
                u2, v2 = center2d + size2d / 2
            else:
                box3d = np.array([*center3d, *size3d, ry], dtype=np.float32)
                box3d_lidar = boxes3d_camera_to_lidar(box3d.reshape(-1, 7), calib).squeeze()
                corners3d = box3d_lidar_to_corners3d(box3d_lidar)  # [8, 3]
                corners_img, _ = calib.lidar_to_img(corners3d)
                u1, v1 = corners_img[:, 0].min(), corners_img[:, 1].min()
                u2, v2 = corners_img[:, 0].max(), corners_img[:, 1].max()

            u1, v1 = max(0, u1), max(0, v1)
            u2, v2 = min(img_size[0] - 1, u2), min(img_size[1] - 1, v2)
            box2d = [u1, v1, u2, v2]

            det_per_img.append([int(cls_id), alpha, *box2d, *size3d, *loc, ry, score])

        selected_indices = nms(det_per_img, calib, nms_thresh)
        det[img_id] = [obj for idx, obj in enumerate(det_per_img) if idx in selected_indices]

    return det
