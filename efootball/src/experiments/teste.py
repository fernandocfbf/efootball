import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

from efootball.src.classes.models.narya_field_homography.KeypointDetectorModel import KeypointDetectorModel
from efootball.src.constants.models import NARYA_KEYPOINT_DETECTOR, NARYA_WEIGTHS_NAME, NARYA_WEIGHTS_TOTAR

kp_model = KeypointDetectorModel(
    backbone='efficientnetb3', num_classes=29, input_shape=(320, 320),
)
checkpoints = tf.keras.utils.get_file(NARYA_WEIGTHS_NAME, NARYA_KEYPOINT_DETECTOR, NARYA_WEIGHTS_TOTAR)
kp_model.load_weights(checkpoints)
test_field_image = cv2.imread('C:/Users/ferna/OneDrive/Documentos/Insper/Efootball/data/frames/0.png')
test_field_image = cv2.cvtColor(test_field_image, cv2.COLOR_BGR2RGB)
pr_mask = kp_model(test_field_image)
print(pr_mask)