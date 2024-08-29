import numpy as np
import cv2
import segmentation_models as sm

from efootball.src.constants.keypoints_detector import INIT_HOMO_MAPPER

def collinear(p0, p1, p2, epsilon=0.001):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < epsilon

def warp_image(img, H, out_shape=None):
    if out_shape is None:
        out_shape = img.shape[-3:-1] if len(img.shape) == 4 else img.shape[:-1]
    if len(img.shape) == 3:
        return cv2.warpPerspective(img, H, dsize=out_shape)
    else:
        if img.shape[0] != H.shape[0]:
            raise ValueError(
                "batch size of images ({}) do not match the batch size of homographies ({})".format(
                    img.shape[0], H.shape[0]
                )
            )
        out_img = []
        for img_, H_ in zip(img, H):
            out_img.append(cv2.warpPerspective(img_, H_, dsize=out_shape))
        return np.array(out_img)

def get_keypoints_from_mask(mask, treshold=0.9):
    keypoints = {}
    indexes = np.argwhere(mask[:, :, :-1] > treshold)
    for indx in indexes:
        id_kp = indx[2]
        if id_kp in keypoints.keys():
            keypoints[id_kp][0].append(indx[0])
            keypoints[id_kp][1].append(indx[1])
        else:
            keypoints[id_kp] = [[indx[0]], [indx[1]]]

    for id_kp in keypoints.keys():
        mean_x = np.mean(np.array(keypoints[id_kp][0]))
        mean_y = np.mean(np.array(keypoints[id_kp][1]))
        keypoints[id_kp] = [mean_y, mean_x]
    return keypoints

def points_from_mask(mask, treshold=0.9):
    list_ids = list()
    src_pts = list()
    dst_pts = list()
    available_keypoints = get_keypoints_from_mask(mask, treshold)
    for id_kp, v in available_keypoints.items():
        src_pts.append(v)
        dst_pts.append(INIT_HOMO_MAPPER[id_kp])
        list_ids.append(id_kp)
    src, dst = np.array(src_pts), np.array(dst_pts)

    ### Final test : return nothing if 3 points are colinear and the src has just 4 points 
    test_colinear = False
    if len(src) == 4:
        if collinear(dst_pts[0], dst_pts[1], dst_pts[2]) or collinear(dst_pts[0], dst_pts[1], dst_pts[3]) or collinear(dst_pts[1], dst_pts[2], dst_pts[3]) :
          test_colinear = True
    src = np.array([]) if test_colinear else src
    dst = np.array([]) if test_colinear else dst
    
    return src, dst

def get_perspective_transform(src, dst):
    if len(src.shape) == 2:
        M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    else:
        M = []
        for src_, dst_ in zip(src, dst):
            M.append(cv2.findHomography(src_, dst_, cv2.RANSAC, 5)[0])
        M = np.array(M)
    return M

def merge_template(img, warped_template):
    valid_index = warped_template[:, :, 0] > 0.0
    overlay = (
        img[valid_index].astype("float32")
        + warped_template[valid_index].astype("float32")
    ) / 2
    new_image = np.copy(img)
    new_image[valid_index] = overlay
    return new_image

def rgb_template_to_coord_conv_template(rgb_template):
    assert isinstance(rgb_template, np.ndarray)
    assert rgb_template.min() >= 0.0
    assert rgb_template.max() <= 1.0
    rgb_template = np.mean(rgb_template, 2)
    x_coord, y_coord = np.meshgrid(
        np.linspace(0, 1, num=rgb_template.shape[1]),
        np.linspace(0, 1, num=rgb_template.shape[0]),
    )
    coord_conv_template = np.stack((rgb_template, x_coord, y_coord), axis=2)
    return coord_conv_template

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
