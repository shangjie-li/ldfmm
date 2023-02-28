import os
import argparse
import yaml
import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt  # for WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data.kitti_dataset import KITTIDataset
from ldfmm import build_model
from helpers.checkpoint_helper import load_checkpoint
from utils.decode_utils import decode_detections
from dataset_player import visualize


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/ldfmm.yaml',
                        help='path to the config file')
    parser.add_argument('--split', type=str, default=None,
                        help='must be chosen from ["train", "val", "trainval", "test"]')
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='score threshold for filtering detections')
    parser.add_argument('--nms_thresh', type=float, default=None,
                        help='NMS threshold for filtering detections')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to the checkpoint')
    parser.add_argument('--show_boxes2d', action='store_true', default=False,
                        help='whether to show 2D boxes')
    parser.add_argument('--show_boxes3d', action='store_true', default=False,
                        help='whether to show 3D boxes')
    parser.add_argument('--show_lidar_points', action='store_true', default=False,
                        help='whether to show lidar point clouds')
    parser.add_argument('--sample_idx', type=str, default=None,
                        help='index of the sample')
    args = parser.parse_args()
    return args


def run(model, dataset, args, cfg, img, lidar_projection_map, info, device):
    inputs = torch.from_numpy(img).unsqueeze(0).to(device)
    lidar_maps = torch.from_numpy(lidar_projection_map).unsqueeze(0).to(device)

    outputs = model(inputs)

    preds = model.select_outputs(outputs, dataset.max_objs, lidar_maps)
    preds = {key: val.detach().cpu().numpy() for key, val in preds.items()}

    img_id = info['img_id']
    img_size = info['img_size']
    infos = {key: np.array(val)[None, ...] for key, val in info.items()}
    calibs = [dataset.get_calib(img_id)]

    det = decode_detections(
        preds=preds,
        infos=infos,
        calibs=calibs,
        regress_box2d=model.regress_box2d,
        score_thresh=cfg['tester']['score_thresh'],
        nms_thresh=cfg['tester']['nms_thresh'],
    )

    objects = det[img_id]
    num_objs = len(objects)
    names = []
    boxes2d = np.zeros((num_objs, 4), dtype=np.float32)
    boxes3d = np.zeros((num_objs, 7), dtype=np.float32)

    for k in range(num_objs):
        obj = objects[k]
        names.append(dataset.class_names[obj[0]])

        uvuv = obj[2:6]
        cu, cv = (uvuv[0] + uvuv[2]) / 2, (uvuv[1] + uvuv[3]) / 2
        width, height = uvuv[2] - uvuv[0], uvuv[3] - uvuv[1]
        boxes2d[k] = np.array([cu, cv, width, height], dtype=np.float32)

        size3d, loc, ry = obj[6:9], obj[9:12], obj[12]
        center3d = np.array(loc) + [0, -size3d[0] / 2, 0]
        boxes3d[k] = np.array([*center3d, *size3d, ry], dtype=np.float32)

    pts = lidar_projection_map.transpose(1, 2, 0).reshape(-1, 3)
    img = img.transpose(1, 2, 0) * dataset.std + dataset.mean
    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_NEAREST)[:, :, ::-1]  # BGR image
    lpm = lidar_projection_map.transpose(1, 2, 0)  # (x, y, z) values in camera coordinates
    lpm = cv2.resize(lpm, dsize=img_size, interpolation=cv2.INTER_NEAREST)

    visualize(
        dataset, args, img, lpm, pts, img_id,
        boxes2d=boxes2d, boxes3d=boxes3d, names=names,
    )


if __name__ == '__main__':
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if args.split is not None:
        cfg['tester']['split'] = args.split
    if args.score_thresh is not None:
        cfg['tester']['score_thresh'] = args.score_thresh
    if args.nms_thresh is not None:
        cfg['tester']['nms_thresh'] = args.nms_thresh
    if args.checkpoint is not None:
        cfg['tester']['checkpoint'] = args.checkpoint

    if cfg['dataset']['type'] == 'KITTI':
        dataset = KITTIDataset(cfg['dataset'], split=cfg['tester']['split'], augment_data=False)
    else:
        raise NotImplementedError

    num_classes = len(cfg['dataset']['class_names'])
    model = build_model(cfg['model'], num_classes)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    assert os.path.exists(cfg['tester']['checkpoint'])
    load_checkpoint(
        file_name=cfg['tester']['checkpoint'],
        model=model,
        optimizer=None,
        map_location=device,
        logger=None,
    )

    torch.set_grad_enabled(False)
    model.eval()

    if args.sample_idx is not None:
        assert args.sample_idx in dataset.id_list
        i = dataset.id_list.index(args.sample_idx)
        img, _, info, lidar_projection_map = dataset[i]
        run(model, dataset, args, cfg, img, lidar_projection_map, info, device)
    else:
        progress_bar = tqdm.tqdm(total=len(dataset), dynamic_ncols=True, leave=True, desc='samples')
        for i in range(len(dataset)):
            img, _, info, lidar_projection_map = dataset[i]
            run(model, dataset, args, cfg, img, lidar_projection_map, info, device)
            progress_bar.update()
        progress_bar.close()
