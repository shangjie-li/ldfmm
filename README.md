# ldfmm

Implementation of LDFMM in PyTorch for KITTI 3D Object Detetcion

## Acknowledgement
 - This repository is developed based on [xinzhuma](https://github.com/xinzhuma/monodle) and [open-mmlab](https://github.com/open-mmlab/OpenPCDet)'s work.

## Installation
 - Clone this repository
   ```
   git clone git@github.com:shangjie-li/ldfmm.git
   ```
 - Install PyTorch environment with Anaconda (Tested on Ubuntu 16.04 & CUDA 10.2)
   ```
   conda create -n ldfmm python=3.6
   conda activate ldfmm
   cd ldfmm
   pip install -r requirements.txt
   ```
 - Compile external modules
   ```
   cd ldfmm
   python setup.py develop
   ```

## KITTI3D Dataset (41.5GB)
 - Download KITTI3D dataset: [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip), [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip), [label_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip) and [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip).
 - Organize the downloaded files as follows
   ```
   ldfmm
   ├── data
   │   ├── kitti
   │   │   │── ImageSets
   │   │   │   ├──test.txt & train.txt & trainval.txt & val.txt
   │   │   │── training
   │   │   │   ├──calib & velodyne & label_2 & image_2
   │   │   │── testing
   │   │   │   ├──calib & velodyne & image_2
   ├── helpers
   ├── layers
   ├── ops
   ├── utils
   ```
 - Display the dataset
   ```
   # Display the dataset and show annotations in the image
   python dataset_player.py --augment_data --show_keypoints --show_boxes2d --show_boxes3d
   
   # Display the dataset and show annotations in point clouds
   python dataset_player.py --augment_data --show_boxes3d --show_lidar_points
   
   # Display the dataset and show the encoded heatmap
   python dataset_player.py --augment_data --show_heatmap
   ```

## Demo
 - Run the demo with a trained model
   ```
   # Show detections in the image
   python demo.py --checkpoint=checkpoints/checkpoint_epoch_80.pth --show_boxes3d
   
   # Show detections in point clouds
   python demo.py --checkpoint=checkpoints/checkpoint_epoch_80.pth --show_boxes3d --show_lidar_points
   ```

## Training
 - Train your model using the following commands
   ```
   python train.py
   ```

## Evaluation
 - Evaluate your model using the following commands
   ```
   python test.py
   ```

