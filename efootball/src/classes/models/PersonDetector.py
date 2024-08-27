import torch
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class PersonDetector():
    def __init__(self, threshold):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.CLASS_INDICES = [0]
        cfg.CLASS_STRING = ['person']
        self.cfg = cfg 
        self.predictor = DefaultPredictor(cfg)
    
    def detect_persons(self, image):
        outputs = self.predictor(image)
        instances = outputs['instances']
        labels = instances.pred_classes
        scores = instances.scores
        masks = instances.pred_masks
        boxes = instances.pred_boxes
        
        # Filter instances to only include "person" class
        person_indices = [i for i, label in enumerate(labels) if label == self.cfg.CLASS_INDICES[0]]
        filtered_boxes = boxes[person_indices].tensor
        filtered_labels = torch.tensor([labels[i] for i in person_indices])
        filtered_scores = torch.tensor([scores[i] for i in person_indices])
        filtered_masks = torch.stack([masks[i] for i in person_indices])
        
        return {"boxes": filtered_boxes, "labels": filtered_labels, "scores": filtered_scores, "masks": filtered_masks}