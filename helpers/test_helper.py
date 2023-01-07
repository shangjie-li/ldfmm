import os
import tqdm
import torch

from helpers.checkpoint_helper import load_checkpoint
from utils.decode_utils import decode_detections


class Tester(object):
    def __init__(self, cfg, model, dataloader, result_dir, logger):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs
        self.class_names = dataloader.dataset.class_names
        self.result_dir = result_dir
        self.logger = logger
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def test(self):
        assert os.path.exists(self.cfg['checkpoint'])
        load_checkpoint(
            file_name=self.cfg['checkpoint'],
            model=self.model,
            optimizer=None,
            map_location=self.device,
            logger=self.logger
        )

        torch.set_grad_enabled(False)
        self.model.eval()
        all_det = {}
        progress_bar = tqdm.tqdm(
            total=len(self.dataloader), dynamic_ncols=True, leave=True, desc='batches'
        )

        for batch_idx, (inputs, targets, infos, lidar_maps) in enumerate(self.dataloader):
            inputs = inputs.to(self.device)
            lidar_maps = lidar_maps.to(self.device)

            outputs = self.model(inputs)

            preds = self.model.select_outputs(outputs, self.max_objs, lidar_maps)
            preds = {key: val.detach().cpu().numpy() for key, val in preds.items()}
            infos = {key: val.detach().cpu().numpy() for key, val in infos.items()}
            calibs = [self.dataloader.dataset.get_calib(index) for index in infos['img_id']]

            det = decode_detections(
                preds=preds,
                infos=infos,
                calibs=calibs,
                score_thresh=self.cfg['score_thresh'],
            )
            all_det.update(det)

            progress_bar.update()
        progress_bar.close()

        self.logger.info('==> Saving results...')
        self.save_results(all_det, self.result_dir)
        self.logger.info('==> Done.')

        self.evaluate(self.result_dir)

    def save_results(self, all_det, result_dir):
        os.makedirs(result_dir, exist_ok=True)
        for img_id in all_det.keys():
            output_path = os.path.join(result_dir, '{:06d}.txt'.format(img_id))
            f = open(output_path, 'w')
            objs = all_det[img_id]
            for i in range(len(objs)):
                class_name = self.class_names[int(objs[i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, 14):
                    f.write(' {:.2f}'.format(objs[i][j]))
                f.write('\n')
            f.close()

    def evaluate(self, result_dir):
        self.dataloader.dataset.eval(result_dir=result_dir, logger=self.logger)
