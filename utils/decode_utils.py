import numpy as np


def decode_detections(preds, infos, calibs, score_thresh):
    det = {}
    batch_size, K, _ = preds['cls_id'].shape
    for i in range(batch_size):
        img_id = infos['img_id'][i]
        downsample = infos['original_downsample'][i]
        calib = calibs[i]
        det_per_img = []
        for j in range(K):
            cls_id = preds['cls_id'][i, j, 0]
            score = preds['score'][i, j, 0]
            center2d = preds['center2d'][i, j, :]
            size2d = preds['size2d'][i, j, :]
            center3d_proj = preds['center3d_proj'][i, j, :]
            depth = preds['depth'][i, j, 0]
            size3d = preds['size3d'][i, j, :]
            alpha_bin = preds['alpha_bin'][i, j, :]
            alpha_res = preds['alpha_res'][i, j, :]

            if score < score_thresh:
                continue

            center2d *= downsample
            size2d *= downsample
            u1, v1 = center2d - size2d / 2
            u2, v2 = center2d + size2d / 2
            box2d = [u1, v1, u2, v2]

            x_proj, y_proj = center3d_proj * downsample
            center3d = calib.img_to_rect(x_proj, y_proj, depth).reshape(-1)
            loc = center3d + [0, size3d[0] / 2, 0]

            bin_id = np.argmax(alpha_bin)
            residual_angle = alpha_res[bin_id]
            alpha = bin_to_angle(bin_id, residual_angle)
            ry = alpha + np.arctan2(center3d[0], center3d[2])

            det_per_img.append([int(cls_id), alpha, *box2d, *size3d, *loc, ry, score])
        det[img_id] = det_per_img

    return det


def normalize_angle(angle):
    sina = np.sin(angle)
    cosa = np.cos(angle)

    return np.arctan2(sina, cosa)


def bin_to_angle(bin_id, residual_angle):
    angle_per_bin = 2 * np.pi / 12
    angle_center = bin_id * angle_per_bin
    angle = angle_center + residual_angle

    return normalize_angle(angle)
