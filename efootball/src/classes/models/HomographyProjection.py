import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

from efootball.src.utils.homography import merge_template, points_from_mask, get_perspective_transform
from efootball.src.utils.homography import warp_image, rgb_template_to_coord_conv_template, build_keypoint_preprocessing

from efootball.src.constants.models import NARYA_KEYPOINT_DETECTOR, NARYA_WEIGTHS_NAME, NARYA_WEIGHTS_TOTAR

class HomographyProjection():
    def __init__(self, backbone:str, num_classes:int, input_shape:tuple):
        self.backbone = backbone
        self.classes = [str(i) for i in range(num_classes)] + ["background"]
        self.input_shape = input_shape

        self.model = sm.FPN(
            self.backbone,
            classes=len(self.classes),
            activation="softmax",
            input_shape=(input_shape[0], input_shape[1], 3),
            encoder_weights="imagenet",
        )
        checkpoints = tf.keras.utils.get_file(NARYA_WEIGTHS_NAME, NARYA_KEYPOINT_DETECTOR, NARYA_WEIGHTS_TOTAR)
        self.model.load_weights(checkpoints)

        template = cv2.imread('C:/Users/ferna/OneDrive/Documentos/Insper/Efootball/efootball/src/img/football_field.png')
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        self.template = cv2.resize(template, (1280,720))/255.
        self.preprocessing = build_keypoint_preprocessing(input_shape, backbone)

    def visualize(self, image):
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.show()

    def get_homography_prediction(self, image):
        image_preprocessed = self.preprocessing(image)
        pr_mask = self.model.predict(np.array([image_preprocessed]))
        src, dst = points_from_mask(pr_mask[0])
        pred_homo = get_perspective_transform(dst,src)
        return pred_homo

    def get_field_map_with_players(self, image, points, colors):
        prediction = self.get_homography_prediction(image)
        inverted_pred_homo = np.linalg.inv(prediction)
        person_points_test = np.array(points)
        person_points_test = person_points_test.astype(np.float32)
        pt_transformed = cv2.perspectiveTransform(person_points_test.reshape(-1, 1, 2), inverted_pred_homo)
        perspective_transformed = pt_transformed.reshape(-1, 2)
        template = self.template.copy()
        template = cv2.resize(template, (320, 320))
        for point, color in zip(perspective_transformed, colors):
            cv2.circle(template, (int(point[0]), int(point[1])), radius=5, color=color, thickness=2)
        return template