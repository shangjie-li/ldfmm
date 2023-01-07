import os
import numpy as np
import argparse
import yaml

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
    parser.add_argument('--cfg_file', type=str, default='data/configs/centernet3d.yaml',
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
    args = parser.parse_args()
    return args


def main():
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    dataset = KITTIDataset(cfg['dataset'], split=args.split, augment_data=args.augment_data)

    for i in range(len(dataset)):
        img, target, info, lidar_projection_map = dataset[i]
        calib = dataset.get_calib(info['img_id'])
        mask_2d = target['mask_2d'].astype(np.bool)
        mask_3d = target['mask_3d'].astype(np.bool)
        keypoints = target['keypoint'][mask_2d]  # (u, v) of keypoints
        boxes2d = target['box2d'][mask_2d]  # (cu, cv, width, height) of bounding boxes
        boxes3d = target['box3d'][mask_3d]  # (x, y, z, h, w, l, ry) in camera coordinates
        flip_flag = target['flip_flag'][mask_3d]

        if args.show_lidar_points:
            boxes3d_lidar = boxes3d_camera_to_lidar(boxes3d, calib)
            pts = lidar_projection_map.transpose(1, 2, 0).reshape(-1, 3)
            pts_lidar = calib.rect_to_lidar(pts)
            open3d_vis_utils.draw_scenes(
                pts_lidar,
                boxes3d_lidar=boxes3d_lidar if args.show_boxes3d else None,
            )
        else:
            img = img.transpose(1, 2, 0) * dataset.std + dataset.mean
            img = img[:, :, ::-1]  # BGR image
            lidar_projection_map = lidar_projection_map.transpose(1, 2, 0)  # (x, y, z) values in camera coordinates
            lidar_projection_map = lidar_projection_map[:, :, 0:3]
            lidar_projection_map = cv2.resize(lidar_projection_map, dsize=dataset.resolution)
            keypoints *= dataset.downsample
            boxes2d *= dataset.downsample
            opencv_vis_utils.draw_scenes(
                img,
                calib,
                keypoints=keypoints if args.show_keypoints else None,
                boxes2d=boxes2d if args.show_boxes2d else None,
                boxes3d_camera=boxes3d if args.show_boxes3d else None,
                flip_flag=flip_flag,
                info=info,
                window_name='img',
                wait_key=False,
            )
            opencv_vis_utils.draw_scenes(
                lidar_projection_map,
                calib,
                keypoints=keypoints if args.show_keypoints else None,
                boxes2d=boxes2d if args.show_boxes2d else None,
                boxes3d_camera=boxes3d if args.show_boxes3d else None,
                flip_flag=flip_flag,
                info=info,
                window_name='lidar_projection_map',
                wait_key=False,
            )
            cv2.waitKey()


if __name__ == '__main__':
    main()
