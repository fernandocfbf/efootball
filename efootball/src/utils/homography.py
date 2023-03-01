import numpy as np
import cv2

from efootball.src.classes.models.narya_field_homography.utils.image import denormalize
from efootball.src.classes.models.narya_field_homography.utils.visualization import visualize, rgb_template_to_coord_conv_template, merge_template
from efootball.src.classes.models.narya_field_homography.utils.masks import points_from_mask
from efootball.src.classes.models.narya_field_homography.utils.homography import get_perspective_transform, warp_image

def get_only_player_coordinates(players_points_information):
    return [point_information[0] for point_information in players_points_information]

def format_points_list(points_list, shape):
    points_formated = list()
    for coordinates in points_list:
        coordinates_resized = np.array([coordinates[0]/shape[0], coordinates[1]/shape[1]])
        points_formated.append(coordinates_resized)
    return points_formated

def homography_player_estimation(key_points_mask, players_points_information, field_shape):
    players_coordinates = get_only_player_coordinates(players_points_information)
    players_coordinates_resized = format_points_list(players_coordinates, field_shape)
    src,dst = points_from_mask(key_points_mask[0])
    
    if len(src) < 4:
        print("Not enough points to estimate homography")
        return []
        
    predicted_homography = get_perspective_transform(dst,src)
    #once the predicted_homography is to plot the 2d image on the 3d field
    #we need to invert it to plot the 3d field on the 2d image
    inverted_homography = np.linalg.inv(predicted_homography)
    players_coordinates_resized = np.array(players_coordinates_resized)
    players_coordinates_resized.astype(np.float32)
    pt_transformed = cv2.perspectiveTransform(
        players_coordinates_resized.reshape(-1, 1, 2),
        inverted_homography
    )
    pt_transformed_to_list = pt_transformed.reshape(-1, 2)
    return pt_transformed_to_list
