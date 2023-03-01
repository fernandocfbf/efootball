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
        self.predictor = DefaultPredictor(cfg)
    
    def detectPersons(self, image):
        outputs = self.predictor(image)
        labels = outputs['instances'].pred_classes
        scores = outputs["instances"].scores
        masks = outputs['instances'].pred_masks
        boxes = outputs['instances'].pred_boxes
        return {"boxes": boxes, "labels":labels, "scores": scores, "masks":masks}