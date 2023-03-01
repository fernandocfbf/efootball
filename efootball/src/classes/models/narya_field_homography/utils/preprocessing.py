import segmentation_models as sm
import numpy as np
import cv2

def build_keypoint_preprocessing(input_shape, backbone):
    sm_preprocessing = sm.get_preprocessing(backbone)
    def preprocessing(input_img, **kwargs):
        to_normalize = False if np.percentile(input_img, 98) > 1.0 else True
        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0] * 255.0 if to_normalize else input_img[0] * 1.0
        else:
            image = input_img * 255.0 if to_normalize else input_img * 1.0

        image = cv2.resize(image, input_shape)
        image = sm_preprocessing(image)
        return image

    return preprocessing