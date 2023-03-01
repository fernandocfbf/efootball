import numpy as np
import cv2

from efootball.src.utils.geometry import define_center_point

def define_player_color(array):
    transpose_matriz = array.T
    red = np.mean(transpose_matriz[0])
    blue = np.mean(transpose_matriz[1])
    green = np.mean(transpose_matriz[2])
    return [red, blue, green]

def define_player_centroid(frame, item_mask):
    segmentation = np.where(item_mask == True)
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))
    kernel = (item_mask * 255).astype('uint8')
    result = cv2.bitwise_and(frame, frame, mask=kernel)
    result_reshape = result.reshape((result.shape[0] * result.shape[1], 3))
    only_player_pixels = result_reshape[result_reshape.sum(axis=(1)) != 0]
    player_centroid = define_player_color(only_player_pixels)
    box = [y_min, y_max, x_min, x_max]
    return box, player_centroid

def get_player_and_ball_informations(image, person_detections, ball_detections, teams_kmeans):
    players_information = list()
    ball_information = list()
    for mask, label in zip(np.asarray(person_detections["masks"].to("cpu")), person_detections['labels']):
            if label == 0:
                box_person, player_color = define_player_centroid(image, mask)
                is_player, team_color = teams_kmeans.indentify_and_predict([player_color])
                if is_player:
                    players_information.append([box_person, team_color])
    
    for box_ball in ball_detections['boxes']:
        ball_information = define_center_point(box_ball[0], box_ball[2], box_ball[1], box_ball[3])
    return players_information, ball_information