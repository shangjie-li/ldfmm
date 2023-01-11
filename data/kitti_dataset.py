import os
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import cv2

from data.kitti_object_eval_python.kitti_common import get_label_annos
from data.kitti_object_eval_python.eval import get_official_eval_result
from utils.encode_utils import angle_to_bin
from utils.encode_utils import gaussian_radius
from utils.encode_utils import draw_umich_gaussian
from utils.affine_utils import get_affine_mat
from utils.affine_utils import affine_transform
from utils.kitti_object3d_utils import parse_objects
from utils.kitti_calibration_utils import parse_calib
from utils.point_clouds_utils import get_completed_lidar_projection_map


class KITTIDataset(Dataset):
    def __init__(self, cfg, split, augment_data=True):
        self.root_dir = 'data/kitti'
        self.split = split
        self.class_names = cfg['class_names']
        self.num_classes = len(self.class_names)
        self.cls_to_id = {}
        for i, name in enumerate(self.class_names):
            self.cls_to_id[name] = i
        self.resolution = np.array([1280, 384])
        self.max_objs = 50
        self.write_list = cfg['write_list']

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
        self.min_distance = cfg['min_distance']
        self.max_distance = cfg['max_distance']
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.downsample = 4

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
        lidar_projection_map = get_completed_lidar_projection_map(pts[:, :3], img, calib)

        h, w, _ = img.shape
        img_size = np.array([w, h])
        feature_size = self.resolution // self.downsample  # W and H

        center = img_size / 2
        aug_size = img_size
        random_flip_flag, random_crop_flag = False, False

        if self.augment_data:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img[:, ::-1, :]
                lidar_projection_map = lidar_projection_map[:, ::-1, :]
                lidar_projection_map[:, :, 0] *= -1

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                aug_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                aug_size = img_size * aug_scale
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # affine_mat: ndarray of float, [2, 3]
        affine_mat = get_affine_mat(center, aug_size, self.resolution)
        img = cv2.warpAffine(
            img, M=affine_mat, dsize=self.resolution, flags=cv2.INTER_NEAREST)
        lidar_projection_map = cv2.warpAffine(
            lidar_projection_map, M=affine_mat, dsize=self.resolution, flags=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # [C, H, W]
        lidar_projection_map = cv2.resize(lidar_projection_map, feature_size, interpolation=cv2.INTER_NEAREST)
        lidar_projection_map = lidar_projection_map.transpose(2, 0, 1)  # [C, H, W]

        info = {
            'img_id': img_id,
            'img_size': img_size,
            'original_downsample': img_size / feature_size,
            'affine_mat': affine_mat,
        }

        if self.split == 'test':
            return img, None, info, lidar_projection_map

        objects = self.get_label(img_id)
        target = {
            'heatmap': np.zeros((self.num_classes, feature_size[1], feature_size[0]), dtype=np.float32),
            'keypoint': np.zeros((self.max_objs, 2), dtype=np.int64),  # (u, v)
            'offset2d': np.zeros((self.max_objs, 2), dtype=np.float32),  # (du, dv)
            'box2d': np.zeros((self.max_objs, 4), dtype=np.float32),  # (cu, cv, width, height)
            'offset3d': np.zeros((self.max_objs, 2), dtype=np.float32),  # (du, dv)
            'box3d': np.zeros((self.max_objs, 7), dtype=np.float32),  # (x, y, z, h, w, l, ry)
            'alpha_bin': np.zeros((self.max_objs, 1), dtype=np.int64),
            'alpha_res': np.zeros((self.max_objs, 1), dtype=np.float32),
            'flip_flag': np.zeros((self.max_objs,), dtype=np.uint8),
            'crop_flag': np.zeros((self.max_objs,), dtype=np.uint8),
            'cls_id': np.zeros((self.max_objs,), dtype=np.int64) - 1,
            'mask': np.zeros((self.max_objs,), dtype=np.int64),
        }

        num_objs = len(objects) if len(objects) <= self.max_objs else self.max_objs
        for i in range(num_objs):
            obj = objects[i]
            if obj.cls_type not in self.write_list: continue
            if obj.level_str == 'UnKnown': continue
            if obj.loc[-1] < self.min_distance or obj.loc[-1] > self.max_distance: continue

            box2d = obj.box2d.copy()  # (u1, v1, u2, v2)
            if random_flip_flag:
                box2d[0], box2d[2] = img_size[0] - box2d[2], img_size[0] - box2d[0]
            box2d[:2] = affine_transform(box2d[:2].reshape(-1, 2), affine_mat).squeeze()
            box2d[2:] = affine_transform(box2d[2:].reshape(-1, 2), affine_mat).squeeze()
            box2d /= self.downsample
            center2d = np.array([(box2d[0] + box2d[2]) / 2, (box2d[1] + box2d[3]) / 2], dtype=np.float32)
            size2d = np.array([box2d[2] - box2d[0], box2d[3] - box2d[1]], dtype=np.float32)
            box2d = np.array([*center2d, *size2d], dtype=np.float32)

            center3d = obj.loc + [0, -obj.h / 2, 0]
            size3d = np.array([obj.h, obj.w, obj.l], dtype=np.float32)
            alpha = obj.alpha
            ry = obj.ry
            center3d_img, _ = calib.rect_to_img(center3d.reshape(-1, 3))
            center3d_img = center3d_img.squeeze()
            if random_flip_flag:
                center3d_img[0] = img_size[0] - center3d_img[0]
                center3d[0] *= -1
                alpha = np.pi - alpha
                ry = np.pi - ry
            box3d = np.array([*center3d, *size3d, ry], dtype=np.float32)
            alpha_bin, alpha_res = angle_to_bin(alpha)
            center3d_img = affine_transform(center3d_img.reshape(-1, 2), affine_mat).squeeze()
            center3d_img /= self.downsample

            keypoint = center3d_img.astype(np.int64)
            if keypoint[0] < 0 or keypoint[0] >= feature_size[0]: continue
            if keypoint[1] < 0 or keypoint[1] >= feature_size[1]: continue
            radius = gaussian_radius(size2d)
            radius = max(0, int(radius))
            cls_id = self.cls_to_id[obj.cls_type]
            draw_umich_gaussian(target['heatmap'][cls_id], keypoint, radius)

            target['keypoint'][i] = keypoint
            target['offset2d'][i] = center2d - keypoint
            target['box2d'][i] = box2d
            target['offset3d'][i] = center3d_img - keypoint
            target['box3d'][i] = box3d
            target['alpha_bin'][i] = alpha_bin
            target['alpha_res'][i] = alpha_res
            target['flip_flag'][i] = random_flip_flag
            target['crop_flag'][i] = random_crop_flag
            target['cls_id'][i] = cls_id
            target['mask'][i] = 1

        if self.split in ['train', 'trainval'] and target['mask'].sum() == 0:
            new_idx = np.random.randint(self.__len__())
            return self.__getitem__(new_idx)

        return img, target, info, lidar_projection_map
