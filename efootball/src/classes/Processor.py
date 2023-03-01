import cv2
import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

from efootball.src.utils.visualization import draw_metricis, draw_bboxes, draw_perspective_field
from efootball.src.utils.players import get_player_and_ball_informations
from efootball.src.utils.homography import homography_player_estimation
from efootball.src.classes.models.narya_field_homography.utils.visualization import rgb_template_to_coord_conv_template

class Processor():
    def __init__(self, person_dector, ball_detector, keypoint_model, ball_metrics, teams_kmeans, args):
        self.person_detector = person_dector
        self.ball_detector = ball_detector
        self.keypoint_model = keypoint_model
        self.ball_metrics = ball_metrics
        self.teams_kmeans = teams_kmeans
        self.args = args
        self.need_to_define_teams_colors = True

        template = cv2.imread('C:/Users/ferna/OneDrive/Documentos/Insper/Efootball/efootball/src/img/football_field.png')
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        template = cv2.resize(template, (320,320))/255
        template = rgb_template_to_coord_conv_template(template)
        self.template = template
    
    def calculate_metrics(self, frame, ball_mask, players_masks, ball_metrics):
        if len(ball_mask["boxes"]) != 0:
            ball_metrics.calculate_possession(frame, ball_mask, players_masks)
        
    def get_video_configs(self):
        cap = cv2.VideoCapture(self.args.path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if self.args.test == "true":
            n_frames = 150
        else:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_cap = cv2.VideoWriter(self.args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        return cap, n_frames, out_cap

    def process_frame(self, frame, out_cap, progress_bar, test=False):
        if self.need_to_define_teams_colors:
            self.teams_kmeans.define_teams_colors(frame, self.person_detector)
            self.ball_metrics.init_possession(self.teams_kmeans.teams)
            self.need_to_define_teams_colors = False

        person_predictions = self.person_detector.detectPersons(frame)
        ball_predictions = self.ball_detector.detectBalls(frame)
        frame_resized = cv2.resize(frame, (320,320))
        keypoints_mask = self.keypoint_model(frame_resized)
        players_info, balls_info = get_player_and_ball_informations(frame, person_predictions, ball_predictions, self.teams_kmeans)
        draw_bboxes(frame, players_info, balls_info)
        self.calculate_metrics(frame, ball_predictions, players_info, self.ball_metrics)
        teams_number = self.teams_kmeans.teams
        draw_metricis(teams_number=teams_number,
            team1_percentage = self.ball_metrics.get_possession_in_percentage(teams_number[0]),
            team2_percentage = self.ball_metrics.get_possession_in_percentage(teams_number[1]),
            frame=frame
        )
        players_points_homography = homography_player_estimation(keypoints_mask, players_info, (320,320))
        draw_perspective_field(frame, self.template, players_points_homography)
        if test:
            plt.imshow(frame)
            plt.show()
        else:
            out_cap.write(frame)
            progress_bar.update(1)
    
    def process(self):
        cap, n_frames, out_cap = self.get_video_configs()
        pbar = tqdm.tqdm(total=n_frames)
        if self.args.test:
            static_frame = cv2.imread('C:/Users/ferna/OneDrive/Documentos/Insper/Efootball/data/frames/90.png')
            self.process_frame(static_frame, None, None, True)
        else:
            for i in range(n_frames):
                _, frame = cap.read()
                self.process_frame(frame, out_cap, pbar)
        pbar.close()
        cap.release()
        out_cap.release()
            