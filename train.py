from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle
from utils import *

# Object detection
config_file_path = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
# config_file_path = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

# # Instance segmentation
# config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

checkpoint_url = config_file_path

output_dir = './output/object_detection'
num_classes = 1

device = 'cuda' # 'cpu'

train_dataset_name = 'LP_train'
train_images_path = 'data/img/train'
train_json_annot_path = 'data/img/train/train.json'

test_dataset_name = 'LP_test'
test_images_path = 'data/img/test'
test_json_annot_path = 'data/img/test/test.json'

cfg_save_path = 'OD_cfg.pickle'

# -- 
register_coco_instances(
    name=train_dataset_name, 
    metadata={},
    json_file=train_json_annot_path, 
    image_root=train_images_path
)
register_coco_instances(
    name=test_dataset_name, 
    metadata={},
    json_file=test_json_annot_path, 
    image_root=test_images_path
)

# plot_samples(dataset_name=train_dataset_name, n=2)

# -- 
def main():
    cfg = get_train_cfg(
        config_file_path, checkpoint_url, 
        train_dataset_name, test_dataset_name, 
        num_classes, device, output_dir
    )
    
    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    trainer.train()
    
if __name__ == '__main__':
    main()

