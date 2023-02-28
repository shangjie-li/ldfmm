import os
import cv2
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset

from data.kitti_object_eval_python.kitti_common import get_label_annos
from data.kitti_object_eval_python.eval import get_official_eval_result
from utils.encode_utils import angle_to_bin
from utils.encode_utils import draw_umich_gaussian
from utils.affine_utils import get_affine_mat
from utils.affine_utils import affine_transform
from utils.kitti_object3d_utils import parse_objects
from utils.kitti_calibration_utils import parse_calib
from utils.point_cloud_utils import get_completed_lidar_projection_map
from utils.box_utils import boxes3d_camera_to_lidar
from ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu


class KITTIDataset(Dataset):
    def __init__(self, cfg, split, augment_data=True):
        self.root_dir = 'data/kitti'
        self.split = split
        self.class_names = cfg['class_names']
        self.num_classes = len(self.class_names)
        self.cls_to_id = {}
        for i, name in enumerate(self.class_names):
            self.cls_to_id[name] = i
        self.write_list = cfg['write_list']
        self.keypoint_encoding = cfg['keypoint_encoding']
        self.depth_diff_thresh = 3.0
        self.max_objs = 50

        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.id_list = [x.strip() for x in open(self.split_file).readlines()]

        self.root_dir = os.path.join(self.root_dir, 'testing' if self.split == 'test' else 'training')
        self.image_dir = os.path.join(self.root_dir, 'image_2')
        self.velodyne_dir = os.path.join(self.root_dir, 'velodyne')
        self.calib_dir = os.path.join(self.root_dir, 'calib')
        self.label_dir = os.path.join(self.root_dir, 'label_2')

        self.augment_data = augment_data
        if self.split not in ['train', 'trainval']:
            self.augment_data = False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.resolution = np.array([1280, 384])
        self.downsample = 4
        self.feature_size = self.resolution // self.downsample

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return io.imread(img_file)  # ndarray of uint8, [H, W, 3], RGB image

    def get_points(self, idx):
        pts_file = os.path.join(self.velodyne_dir, '%06d.bin' % idx)
        assert os.path.exists(pts_file)
        return np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)  # ndarray of float32, [N, 4], (x, y, z, i)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return parse_calib(calib_file)  # kitti_calibration_utils.Calibration

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return parse_objects(label_file)  # list of kitti_object3d_utils.Object3d

    def eval(self, result_dir, logger):
        logger.info('==> Loading detections and ground truths...')
        img_ids = [int(idx) for idx in self.id_list]
        dt_annos = get_label_annos(result_dir)
        gt_annos = get_label_annos(self.label_dir, img_ids)
        logger.info('==> Done.')

        logger.info('==> Evaluating...')
        test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        for category in self.write_list:
            result_str = get_official_eval_result(
                gt_annos, dt_annos, test_id[category], use_ldf_eval=False, print_info=False)
            logger.info(result_str)

    def __len__(self):
        return self.id_list.__len__()

    def __getitem__(self, idx):
        img_id = int(self.id_list[idx])
        img = self.get_image(img_id)
        pts = self.get_points(img_id)
        calib = self.get_calib(img_id)

        # lidar_projection_map: ndarray of float32, [H, W, 3], (x, y, z) values in camera coordinates
        lpm = get_completed_lidar_projection_map(pts[:, :3], img, calib)

        h, w, _ = img.shape
        img_size = np.array([w, h])
        center = img_size / 2
        aug_size = img_size
        random_flip_flag, random_crop_flag = False, False
        if self.augment_data:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img[:, ::-1, :]
                lpm = lpm[:, ::-1, :]
                lpm[:, :, 0] *= -1
            if np.random.random() < self.random_crop:
                random_crop_flag = True
                aug_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                aug_size = img_size * aug_scale
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # affine_mat: ndarray of float, [2, 3]
        affine_mat = get_affine_mat(center, aug_size, self.resolution)

        img = cv2.warpAffine(img, M=affine_mat, dsize=self.resolution, flags=cv2.INTER_NEAREST)
        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # [C, H, W]

        lpm = cv2.warpAffine(lpm, M=affine_mat, dsize=self.resolution, flags=cv2.INTER_NEAREST)
        lpm = cv2.resize(lpm, self.feature_size, interpolation=cv2.INTER_NEAREST)
        lpm = lpm.transpose(2, 0, 1)  # [C, H, W]

        info = {
            'img_id': img_id,
            'img_size': img_size,
            'original_downsample': img_size / self.feature_size,
            'affine_mat': affine_mat,
            'flip_flag': random_flip_flag,
            'crop_flag': random_crop_flag,
        }

        if self.split == 'test':
            return img, None, info, lpm

        objects = self.get_label(img_id)
        target = {
            'heatmap': np.zeros((self.num_classes, self.feature_size[1], self.feature_size[0]), dtype=np.float32),
            'keypoint': np.zeros((self.max_objs, 2), dtype=np.int64),  # (u, v)
            'offset2d': np.zeros((self.max_objs, 2), dtype=np.float32),  # (du, dv)
            'box2d': np.zeros((self.max_objs, 4), dtype=np.float32),  # (cu, cv, width, height)
            'offset3d': np.zeros((self.max_objs, 2), dtype=np.float32),  # (du, dv)
            'box3d': np.zeros((self.max_objs, 7), dtype=np.float32),  # (x, y, z, h, w, l, ry)
            'alpha_bin': np.zeros((self.max_objs, 1), dtype=np.int64),
            'alpha_res': np.zeros((self.max_objs, 1), dtype=np.float32),
            'cls_id': np.zeros((self.max_objs,), dtype=np.int64) - 1,
            'mask': np.zeros((self.max_objs,), dtype=np.int64),
        }

        num_objs = len(objects) if len(objects) <= self.max_objs else self.max_objs
        for i in range(num_objs):
            obj = objects[i]
            if obj.cls_type not in self.class_names: continue

            pts_img_in_box3d = self.obtain_pts_img_in_box3d(obj, info, calib, lpm)
            if pts_img_in_box3d.shape[0] == 0: continue

            if self.keypoint_encoding == 'LidarPoints':
                keypoint = self.obtain_keypoint_by_lidar_points(obj, pts_img_in_box3d, lpm)
            elif self.keypoint_encoding == 'Center3D':
                keypoint = self.obtain_center3d_img(obj, info, calib)
            else:
                raise NotImplementedError

            if keypoint[0] < 0 or keypoint[0] >= self.feature_size[0]: continue
            if keypoint[1] < 0 or keypoint[1] >= self.feature_size[1]: continue

            box2d = self.obtain_box2d(obj, info)
            center2d, size2d = box2d[0:2], box2d[2:4]

            box3d = self.obtain_box3d(obj, info)
            center3d_img = self.obtain_center3d_img(obj, info, calib)
            alpha = self.obtain_alpha(obj, info)
            alpha_bin, alpha_res = angle_to_bin(alpha)

            depth = box3d[2]
            if abs(depth - lpm[-1, keypoint[1], keypoint[0]]) > self.depth_diff_thresh: continue

            cls_id = self.cls_to_id[obj.cls_type]
            radius = int(size2d.min() / 2)
            draw_umich_gaussian(target['heatmap'][cls_id], keypoint, radius)

            target['keypoint'][i] = keypoint
            target['offset2d'][i] = center2d - keypoint
            target['box2d'][i] = box2d
            target['offset3d'][i] = center3d_img - keypoint
            target['box3d'][i] = box3d
            target['alpha_bin'][i] = alpha_bin
            target['alpha_res'][i] = alpha_res
            target['cls_id'][i] = cls_id
            target['mask'][i] = 1

        if self.split in ['train', 'trainval'] and target['mask'].sum() == 0:
            new_idx = np.random.randint(self.__len__())
            return self.__getitem__(new_idx)

        return img, target, info, lpm

    def obtain_box2d(self, obj, info):
        uvuv = obj.box2d.copy()  # (u1, v1, u2, v2)
        if info['flip_flag']:
            uvuv[0], uvuv[2] = info['img_size'][0] - uvuv[2], info['img_size'][0] - uvuv[0]
        uvuv[:2] = affine_transform(uvuv[:2].reshape(-1, 2), info['affine_mat']).squeeze()
        uvuv[2:] = affine_transform(uvuv[2:].reshape(-1, 2), info['affine_mat']).squeeze()
        uvuv /= self.downsample

        center2d = np.array([(uvuv[0] + uvuv[2]) / 2, (uvuv[1] + uvuv[3]) / 2], dtype=np.float32)
        size2d = np.array([uvuv[2] - uvuv[0], uvuv[3] - uvuv[1]], dtype=np.float32)

        return np.array([*center2d, *size2d], dtype=np.float32)

    def obtain_box3d(self, obj, info):
        center3d = obj.loc + [0, -obj.h / 2, 0]
        size3d = np.array([obj.h, obj.w, obj.l], dtype=np.float32)
        ry = obj.ry
        if info['flip_flag']:
            center3d[0] *= -1
            ry = np.pi - ry

        return np.array([*center3d, *size3d, ry], dtype=np.float32)

    def obtain_alpha(self, obj, info):
        alpha = obj.alpha
        if info['flip_flag']:
            alpha = np.pi - alpha

        return alpha

    def obtain_pts_img_in_box3d(self, obj, info, calib, lpm):
        lpm = lpm.copy()
        lpm = lpm.transpose(1, 2, 0)
        if info['flip_flag']:
            lpm[:, :, 0] *= -1
        pts_lidar = calib.rect_to_lidar(lpm.reshape(-1, 3))

        center3d = obj.loc + [0, -obj.h / 2, 0]
        size3d = np.array([obj.h, obj.w, obj.l], dtype=np.float32)
        ry = obj.ry
        boxes_lidar = boxes3d_camera_to_lidar(np.array([*center3d, *size3d, ry]).reshape(-1, 7), calib)

        point_indices = points_in_boxes_cpu(
            torch.from_numpy(pts_lidar),
            torch.from_numpy(boxes_lidar),
        ).numpy()  # [num_boxes, num_points]
        pts_lidar_in_box3d = pts_lidar[point_indices[0] > 0]

        pts_img, _ = calib.lidar_to_img(pts_lidar_in_box3d)
        if info['flip_flag']:
            pts_img[:, 0] = info['img_size'][0] - pts_img[:, 0]
        pts_img = affine_transform(pts_img, info['affine_mat'])
        pts_img /= self.downsample
        pts_img = pts_img.astype(np.int64)

        return pts_img

    def obtain_keypoint_by_lidar_points(self, obj, pts_img, lpm, max_iters=10):
        mean_u, mean_v = pts_img[:, 0].mean(), pts_img[:, 1].mean()
        dis = np.sqrt((pts_img[:, 0] - mean_u) ** 2 + (pts_img[:, 1] - mean_v) ** 2)
        indices = np.argsort(dis)
        pts_img = pts_img[indices] if indices.shape[0] < max_iters else pts_img[indices[:max_iters]]

        depth = obj.loc[-1]
        keypoint = pts_img[0]
        for i in range(pts_img.shape[0]):
            keypoint = pts_img[i]
            if keypoint[0] < 0 or keypoint[0] >= self.feature_size[0]: continue
            if keypoint[1] < 0 or keypoint[1] >= self.feature_size[1]: continue
            if abs(depth - lpm[-1, keypoint[1], keypoint[0]]) <= self.depth_diff_thresh: break

        return keypoint

    def obtain_center3d_img(self, obj, info, calib):
        center3d = obj.loc + [0, -obj.h / 2, 0]
        center3d_img, _ = calib.rect_to_img(center3d.reshape(-1, 3))
        center3d_img = center3d_img.squeeze()
        if info['flip_flag']:
            center3d_img[0] = info['img_size'][0] - center3d_img[0]
        center3d_img = affine_transform(center3d_img.reshape(-1, 2), info['affine_mat']).squeeze()
        center3d_img /= self.downsample
        center3d_img = center3d_img.astype(np.int64)

        return center3d_img
