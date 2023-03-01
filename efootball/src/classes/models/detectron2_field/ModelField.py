from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class FieldDetector():
    def __init__(self, threshold):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 8
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = (32, 64, 128)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.DEVICE='cuda'
        cfg.MODEL.WEIGHTS = "C:/Users/ferna/OneDrive/Documentos/Insper/mmn_train/src/output/model_final.pth"
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.9
        self.predictor = DefaultPredictor(cfg)
    
    def detectField(self, image):
        outputs = self.predictor(image)
        return outputs

    def removeBackground(self, image):
        outputs = self.detectField(image)
        masks = outputs["instances"].pred_masks
        mask = masks.sum(0) > 0
        mask = mask.unsqueeze(0).float()
        mask = mask.cpu().numpy().squeeze()
        masked_img = image.copy()
        masked_img[mask == 0] = [0, 0, 0]
        return masked_img