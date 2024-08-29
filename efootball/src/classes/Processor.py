import cv2
import tqdm
import numpy as np

from efootball.src.classes.models.TeamsSegmentation import TeamsSegmentation
from efootball.src.classes.metrics.BallMetrics import BallMetrics
from efootball.src.classes.models.PersonDetector import PersonDetector
from efootball.src.classes.models.BallDetector import BallDetector
from efootball.src.classes.models.HomographyProjection import HomographyProjection

from efootball.src.utils.geometry import define_center_point
from efootball.src.utils.visualization import draw_based_on_predictions, draw_circle, draw_metricis

from efootball.src.constants.teams import BALL_COLOR

class Processor():
    def __init__(self, args):
        self.args = args

        #models
        self.person_detector = PersonDetector(threshold=0.55)
        self.teams_segmentation = TeamsSegmentation(num_clusters=4)
        self.ball_detector = BallDetector(threshold=0.8)
        self.homography_projection = HomographyProjection(backbone="efficientnetb3", num_classes=29, input_shape=(1080,1920), output_shape=(320,320))

        #metrics
        self.ball_metrics = BallMetrics()

        self.homography_prediction = None
    
    def get_video_configs(self):
        cap = cv2.VideoCapture(self.args.path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_cap = cv2.VideoWriter(self.args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        return cap, n_frames, out_cap

    def process_frame(self, frame, out_cap):
        person_predictions = self.person_detector.detect_persons(frame)
        player_predictions = self.teams_segmentation.get_players(frame, person_predictions)
        ball_predictions = self.ball_detector.detect_balls(frame)
        homography_prediction = self.homography_projection.project_players_position(frame, player_predictions)

        if len(ball_predictions["boxes"]) > 0:
            self.ball_metrics.calculate_possession(frame, ball_predictions, player_predictions)
            ball_position = define_center_point(ball_predictions["boxes"][0])
            draw_circle(frame, ball_position, BALL_COLOR)

        draw_based_on_predictions(frame, player_predictions)
        draw_metricis(frame, self.ball_metrics.get_metrics())
        if homography_prediction is not None:
            self.homography_prediction = homography_prediction
         
        frame[760:1080, 1600:1920] = self.homography_prediction
        out_cap.write(frame)
    
    def process(self):
        cap, n_frames, out_cap = self.get_video_configs()
        progress_bar = tqdm.tqdm(total=n_frames)
        for i in range(n_frames):
            _, frame = cap.read()
            if i == 0:
                self.teams_segmentation.define_teams_colors(frame)
                self.ball_metrics.init_possession(self.teams_segmentation.teams)
            self.process_frame(frame, out_cap)
            progress_bar.update(1)
        progress_bar.close()
        cap.release()
        out_cap.release()
            