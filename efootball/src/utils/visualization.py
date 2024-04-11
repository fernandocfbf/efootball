import cv2
import numpy as np

from efootball.src.utils.geometry import define_center_point
from efootball.src.constants.teams import TEAMS_COLORS_RGB

def draw_bouding_box(image, box_coordiantes, color): 
    cv2.rectangle(image,
                (box_coordiantes[0], box_coordiantes[1]),
                (box_coordiantes[2], box_coordiantes[3]),
                color,
                2
            )

def draw_based_on_predictions(image, predictions):
    for mask, team in zip(predictions["masks"], predictions['teams']):
        segmentation = np.where(mask==True)
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))
        box = [x_min, y_min, x_max, y_max]
        draw_bouding_box(image, box, TEAMS_COLORS_RGB[team]["color_code"])

def draw_metricis(teams_number, team1_percentage, team2_percentage, frame):
        team1 = TEAMS_COLORS_RGB[teams_number[0]]
        team2 = TEAMS_COLORS_RGB[teams_number[1]]
        cv2.putText(
            frame,
            f"{team1['color_name']}: {team1_percentage}%",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            team1["color_code"],
            2
        )
        cv2.putText(
            frame,
            f"{team2['color_name']}: {team2_percentage}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            team2["color_code"],
            2
        )

def draw_bboxes(image, player_informations, ball_information):
    for player_information in player_informations:
        box_person = player_information[0]
        team_color = player_information[1]
        cv2.rectangle(
            image,
            (box_person[2], box_person[0]),
            (box_person[3], box_person[1]),
            TEAMS_COLORS_RGB[team_color]["color_code"],
            2
        )
    if ball_information:
        cv2.circle(image, (ball_information[0], ball_information[1]), radius=12, color=(255, 255, 255), thickness=2)

def draw_perspective_field(image, field_template, player_positions):
    for box_coordinates in player_positions:
        print(box_coordinates)
        box_center_position = define_center_point(
            box_coordinates[0],
            box_coordinates[1],
            box_coordinates[2],
            box_coordinates[3]
        )
        #draw circle using box center position
        cv2.circle(field_template, (int(box_center_position[0]), int(box_center_position[1])), radius=2, color=(255,255,255), thickness=2)
    #paste a field template on the bottom right of the image with 320x320 size
    image[0:320, 0:320] = cv2.resize(field_template, (320, 320))
