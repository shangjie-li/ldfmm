import os
import argparse
import yaml
import tqdm
import numpy as np
import matplotlib.pyplot as plt  # for WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data.kitti_dataset import KITTIDataset
from utils.box_utils import boxes3d_camera_to_lidar
from utils import opencv_vis_utils
from utils import open3d_vis_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/ldfmm.yaml',
                        help='path to the config file')
    parser.add_argument('--split', type=str, default='train',
                        help='must be chosen from ["train", "val", "trainval", "test"]')
    parser.add_argument('--augment_data', action='store_true', default=False,
                        help='whether to use random data augmentation')
    parser.add_argument('--show_keypoints', action='store_true', default=False,
                        help='whether to show keypoints')
    parser.add_argument('--show_boxes2d', action='store_true', default=False,
                        help='whether to show 2D boxes')
    parser.add_argument('--show_boxes3d', action='store_true', default=False,
                        help='whether to show 3D boxes')
    parser.add_argument('--show_lidar_points', action='store_true', default=False,
                        help='whether to show lidar point clouds')
    parser.add_argument('--show_heatmap', action='store_true', default=False,
                        help='whether to show the heatmap')
    parser.add_argument('--sample_idx', type=str, default=None,
                        help='index of the sample')
    args = parser.parse_args()
    return args


def visualize(dataset, args, img, lpm, pts, img_id,
              heatmap=None, keypoints=None, boxes2d=None, boxes3d=None, names=None, info=None):
    calib = dataset.get_calib(img_id)

    img = opencv_vis_utils.normalize_img(img)
    img_window_name = 'img: %06d' % img_id

    valid_mask = (lpm[:, :, 0:1] != 0) | (lpm[:, :, 1:2] != 0) | (lpm[:, :, 2:3] != 0)
    lpm = opencv_vis_utils.normalize_img(lpm)
    lpm = np.zeros_like(lpm) + lpm * valid_mask
    lpm_window_name = 'lidar_projection_map: %06d' % img_id

    if pts is not None and args.show_lidar_points:
        boxes3d_lidar = boxes3d_camera_to_lidar(boxes3d, calib)
        pts_lidar = calib.rect_to_lidar(pts)
        open3d_vis_utils.draw_scene(
            pts_lidar,
            boxes3d_lidar=boxes3d_lidar if boxes3d is not None and args.show_boxes3d else None,
            names=names,
            window_name='%06d' % img_id
        )
        return

    data = {img_window_name: img, lpm_window_name: lpm}
    if heatmap is not None and args.show_heatmap:
        for k, v in data.items():
            data[k] = opencv_vis_utils.draw_heatmap(
                v,
                heatmap,
                dataset.class_names,
            )
            cv2.namedWindow(k, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(k, data[k].shape[1], data[k].shape[0])
            cv2.imshow(k, data[k])

    else:
        for k, v in data.items():
            data[k] = opencv_vis_utils.draw_scene(
                v,
                calib,
                keypoints=keypoints if keypoints is not None and args.show_keypoints else None,
                boxes2d=boxes2d if boxes2d is not None and args.show_boxes2d else None,
                boxes3d_camera=boxes3d if boxes3d is not None and args.show_boxes3d else None,
                names=names,
                info=info,
            )
            cv2.namedWindow(k, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(k, data[k].shape[1], data[k].shape[0])
            cv2.imshow(k, data[k])

    cv2.moveWindow(img_window_name, 0, 0)
    cv2.moveWindow(lpm_window_name, 0, 450)

    key = cv2.waitKey(0)
    while key:
        if key == 27:  # Esc
            cv2.destroyAllWindows()
            break
        elif key == 13:  # Enter
            for k, v in data.items():
                cv2.imwrite(k + '.png', v)
            cv2.destroyAllWindows()
            break
        else:
            key = cv2.waitKey(0)


def run(dataset, args, img, target, info, lidar_projection_map):
    img_id = info['img_id']
    pts = lidar_projection_map.transpose(1, 2, 0).reshape(-1, 3)
    img = img.transpose(1, 2, 0) * dataset.std + dataset.mean
    img = img[:, :, ::-1]  # BGR image in the size of dataset.resolution
    lpm = lidar_projection_map.transpose(1, 2, 0)  # (x, y, z) values in camera coordinates
    lpm = cv2.resize(lpm, dsize=dataset.resolution, interpolation=cv2.INTER_NEAREST)

    if target is None:
        visualize(
            dataset, args, img, lpm, pts, img_id, info=info,
        )

    else:
        mask = target['mask'].astype(np.bool)
        heatmap = target['heatmap']
        keypoints = target['keypoint'][mask]  # (u, v) of keypoints
        boxes2d = target['box2d'][mask]  # (cu, cv, width, height) of bounding boxes
        boxes3d = target['box3d'][mask]  # (x, y, z, h, w, l, ry) in camera coordinates
        cls_ids = target['cls_id'][mask]
        names = [dataset.class_names[idx] for idx in cls_ids]

        heatmap = np.stack([
            cv2.resize(heatmap[k], dsize=dataset.resolution) for k in range(heatmap.shape[0])
        ], axis=0)
        keypoints *= dataset.downsample
        boxes2d *= dataset.downsample

        visualize(
            dataset, args, img, lpm, pts, img_id,
            heatmap=heatmap, keypoints=keypoints, boxes2d=boxes2d, boxes3d=boxes3d, names=names, info=info,
        )


if __name__ == '__main__':
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if cfg['dataset']['type'] == 'KITTI':
        dataset = KITTIDataset(cfg['dataset'], split=args.split, is_training=False, augment_data=args.augment_data)
    else:
        raise NotImplementedError

    if args.sample_idx is not None:
        assert args.sample_idx in dataset.id_list
        i = dataset.id_list.index(args.sample_idx)
        img, target, info, lidar_projection_map = dataset[i]
        run(dataset, args, img, target, info, lidar_projection_map)
    else:
        progress_bar = tqdm.tqdm(total=len(dataset), dynamic_ncols=True, leave=True, desc='samples')
        for i in range(len(dataset)):
            img, target, info, lidar_projection_map = dataset[i]
            run(dataset, args, img, target, info, lidar_projection_map)
            progress_bar.update()
        progress_bar.close()
