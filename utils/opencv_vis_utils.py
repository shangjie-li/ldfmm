import cv2
import numpy as np

from utils.box_utils import boxes3d_camera_to_lidar
from utils.box_utils import box3d_lidar_to_corners3d
from utils.affine_utils import affine_transform

box_colormap = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 255, 0),
    'Cyclist': (0, 255, 255),
}  # BGR


def normalize_img(img):
    """
    Normalize the image data to np.uint8 and image shape to [H, W, 3].

    Args:
        img: ndarray, [H, W] or [H, W, 1] or [H, W, 3]

    Returns:
        img: ndarray of uint8, [H, W, 3]

    """
    img = img.copy()
    img += -img.min() if img.min() < 0 else 0
    img = np.clip(img / img.max(), a_min=0., a_max=1.) * 255.
    img = img.astype(np.uint8)
    img = img[:, :, None] if len(img.shape) == 2 else img

    assert len(img.shape) == 3
    if img.shape[-1] == 1:
        return cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif img.shape[-1] == 3:
        return img
    else:
        raise NotImplementedError


def draw_scene(img, calib, keypoints=None, boxes2d=None, names_2d=None, boxes3d_camera=None, names_3d=None,
               flip_flag=None, info=None, window_name='image', wait_key=True, enter_to_save=True):
    """
    Show the image with 2D boxes or 3D boxes.

    Args:
        img: ndarray of uint8, [H, W, 3], BGR image
        calib: kitti_calibration_utils.Calibration
        keypoints: ndarray of float32, [N, 2], (u, v) of keypoints
        boxes2d: ndarray of float32, [N, 4], (cu, cv, width, height) of bounding boxes
        names_2d: list of str, name of each object
        boxes3d_camera: ndarray of float32, [N, 7], (x, y, z, h, w, l, ry] in camera coordinates
        names_3d: list of str, name of each object
        flip_flag: ndarray of uint8, [N]
        info: dict
        window_name: str
        wait_key: bool
        enter_to_save: bool

    Returns:

    """
    if keypoints is not None:
        img = draw_keypoints(img, keypoints, names_2d)

    if boxes2d is not None:
        img = draw_boxes2d(img, boxes2d, names_2d)

    if boxes3d_camera is not None:
        img = draw_boxes3d(img, calib, boxes3d_camera, names_3d, flip_flag, info)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, img.shape[1], img.shape[0])
    cv2.imshow(window_name, img)

    if wait_key:
        # Press 'Esc' to close this window, or 'Enter' to save the image.
        key = cv2.waitKey(0)
        while key:
            if key == 27:  # Esc
                cv2.destroyWindow(window_name)
                break
            elif enter_to_save and key == 13:  # Enter
                cv2.imwrite(window_name + '.png', img)
                cv2.destroyWindow(window_name)
                break
            else:
                key = cv2.waitKey(0)


def draw_keypoints(img, keypoints, names=None, radius=1, color=(0, 255, 0), thickness=2):
    """
    Draw keypoints in the image.

    Args:
        img: ndarray of uint8, [H, W, 3], BGR image
        keypoints: ndarray of float32, [N, 2], (u, v) of keypoints
        names: list of str, name of each object
        color: tuple
        thickness: int

    Returns:
        img: ndarray of uint8, [H, W, 3], BGR image

    """
    for i in range(keypoints.shape[0]):
        keypoint = keypoints[i]
        u, v = int(keypoint[0]), int(keypoint[1])

        if names is not None:
            color = box_colormap[names[i]]

        img = cv2.circle(img, (u, v), radius=radius, color=color, thickness=thickness)

    return img


def draw_boxes2d(img, boxes2d, names=None, color=(0, 255, 0), thickness=2, show_mask=True):
    """
    Draw 2D boxes in the image.

    Args:
        img: ndarray of uint8, [H, W, 3], BGR image
        boxes2d: ndarray of float32, [N, 4], (cu, cv, width, height) of bounding boxes
        names: list of str, name of each object
        color: tuple
        thickness: int
        show_mask: bool

    Returns:
        img: ndarray of uint8, [H, W, 3], BGR image

    """
    for i in range(boxes2d.shape[0]):
        cu, cv, width, height = boxes2d[i]
        u1, v1 = int(cu - width / 2), int(cv - height / 2)
        u2, v2 = int(cu + width / 2), int(cv + height / 2)

        if names is not None:
            color = box_colormap[names[i]]

        cv2.rectangle(img, (u1, v1), (u2, v2), color, thickness)

        if show_mask:
            mask = np.zeros(img.shape, dtype=np.uint8)
            mask[v1:v2, u1:u2, :] = np.array(color) * 0.25
            img = img.astype(np.int64) + mask.astype(np.int64)
            img = np.clip(img, a_min=0, a_max=255).astype(np.uint8)

    return img


def draw_boxes3d(img, calib, boxes3d_camera, names=None, flip_flag=None, info=None,
                 color=(0, 255, 0), thickness=2):
    """
    Draw 3D boxes in the image.

    Args:
        img: ndarray of uint8, [H, W, 3], BGR image
        calib: kitti_calibration_utils.Calibration
        boxes3d_camera: ndarray of float32, [N, 7], (x, y, z, h, w, l, ry] in camera coordinates
        names: list of str, name of each object
        flip_flag: ndarray of uint8, [N]
        info: dict
        color: tuple
        thickness: int

    Returns:
        img: ndarray of uint8, [H, W, 3], BGR image

    """
    for i in range(boxes3d_camera.shape[0]):
        box3d_camera = boxes3d_camera[i].copy()
        if flip_flag is not None:
            if flip_flag[i]:
                box3d_camera[0] *= -1
                box3d_camera[-1] = np.pi - box3d_camera[-1]
            box3d_lidar = boxes3d_camera_to_lidar(box3d_camera.reshape(-1, 7), calib).squeeze()
            corners3d = box3d_lidar_to_corners3d(box3d_lidar)  # [8, 3]
            corners_img, corners_img_depth = calib.lidar_to_img(corners3d)
            if flip_flag[i]:
                corners_img[:, 0] = info['img_size'][0] - corners_img[:, 0]
            corners_img = affine_transform(corners_img, info['affine_mat'])
        else:
            box3d_lidar = boxes3d_camera_to_lidar(box3d_camera.reshape(-1, 7), calib).squeeze()
            corners3d = box3d_lidar_to_corners3d(box3d_lidar)  # [8, 3]
            corners_img, corners_img_depth = calib.lidar_to_img(corners3d)

        if (corners_img_depth > 0).sum() < 8:
            continue

        if names is not None:
            color = box_colormap[names[i]]

        pts_img = corners_img.astype(np.int)
        cv2.line(img, (pts_img[0, 0], pts_img[0, 1]), (pts_img[5, 0], pts_img[5, 1]), color, thickness)
        cv2.line(img, (pts_img[1, 0], pts_img[1, 1]), (pts_img[4, 0], pts_img[4, 1]), color, thickness)
        for k in range(4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)
            i, j = k, k + 4
            cv2.line(img, (pts_img[i, 0], pts_img[i, 1]), (pts_img[j, 0], pts_img[j, 1]), color, thickness)

    return img


def draw_heatmap(img, heatmap, names, window_name='image', wait_key=True, enter_to_save=True):
    """
    Show the image with the heatmap.

    Args:
        img: ndarray of uint8, [H, W, 3], BGR image
        heatmap: ndarray of float32, [C, H, W], heatmap (0 to 1)
        names: list of str, name corresponding to each channel of the heatmap
        window_name: str
        wait_key: bool
        enter_to_save: bool

    Returns:

    """
    for i in range(heatmap.shape[0]):
        hm = heatmap[i]
        color = box_colormap[names[i]]
        hm = hm[:, :, None].repeat(3, axis=-1) * np.array(color)
        img = img.astype(np.int64) + hm.astype(np.int64)
        img = np.clip(img, a_min=0, a_max=255).astype(np.uint8)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, img.shape[1], img.shape[0])
    cv2.imshow(window_name, img)

    if wait_key:
        # Press 'Esc' to close this window, or 'Enter' to save the image.
        key = cv2.waitKey(0)
        while key:
            if key == 27:  # Esc
                cv2.destroyWindow(window_name)
                break
            elif enter_to_save and key == 13:  # Enter
                cv2.imwrite(window_name + '.png', img)
                cv2.destroyWindow(window_name)
                break
            else:
                key = cv2.waitKey(0)
