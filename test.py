from detectron2.engine import DefaultPredictor

import os
import pickle
from utils import *

cfg_save_path = 'OD_cfg.pickle'

with open(cfg_pave_path, 'rb') as f:
    cfg = pickle.load(f)
    
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = 'test/vechicle with license plate_37.jpeg'
videoPath = 'test/highway.mp4'

on_image(image_path, predictor)
on_video(videoPath, predictor)
