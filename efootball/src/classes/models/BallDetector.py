from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np

class BallDetector():
    def __init__(self, threshold):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "efootball\src\classes\models\detectron2_ball\weights.pth"
        cfg.MODEL.RETINANET.NUM_CLASSES = 2
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold  # set threshold for this model
        cfg.MODEL.DEVICE='cuda'
        self.predictor = DefaultPredictor(cfg)
    
    def detect_balls(self, image):
        outputs = self.predictor(image)
        scores = outputs["instances"].scores.tolist()
        boxes = outputs['instances'].pred_boxes.tensor.tolist()
        if len(boxes) == 0:
            return {"boxes": [], "scores": []}
        top_idx = int(np.argmax(scores))        
        most_sure_instance = boxes[top_idx]
        most_sure_instance = [int(x) for x in most_sure_instance]
        higher_score = scores[top_idx]
        return {"boxes": [most_sure_instance], "scores": [higher_score]}