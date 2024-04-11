import cv2
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt

from efootball.src.classes.models.TeamsSegmentation import TeamsSegmentation
from efootball.src.classes.metrics.ball import BallMetrics
from efootball.src.classes.models.PersonDetector import PersonDetector
from efootball.src.classes.models.BallDetector import BallDetector
from efootball.src.classes.models.narya_field_homography.KeypointDetectorModel import KeypointDetectorModel

from efootball.src.constants.models import NARYA_KEYPOINT_DETECTOR, NARYA_WEIGHTS_TOTAR, NARYA_WEIGTHS_NAME
from efootball.src.utils.visualization import draw_based_on_predictions, draw_metricis, draw_bboxes, draw_perspective_field
from efootball.src.utils.players import get_player_and_ball_informations
from efootball.src.utils.homography import homography_player_estimation
from efootball.src.classes.models.narya_field_homography.utils.visualization import rgb_template_to_coord_conv_template

class Processor():
    def __init__(self, args):
        self.args = args
        self.person_detector = PersonDetector(threshold=0.6)
        self.teams_segmentation = TeamsSegmentation(num_clusters=4)
        self.ball_detector = BallDetector(threshold=0.8)
        #kp_model = KeypointDetectorModel(backbone='efficientnetb3', num_classes=29, input_shape=(320, 320))
        #checkpoints = tf.keras.utils.get_file(NARYA_WEIGTHS_NAME, NARYA_KEYPOINT_DETECTOR, NARYA_WEIGHTS_TOTAR)
        #kp_model.load_weights(checkpoints)
        
        
        #self.keypoint_model = kp_model
        #self.ball_metrics = BallMetrics()
        
        #self.args = args
        #self.need_to_define_teams_colors = True

        #template = cv2.imread('C:/Users/ferna/OneDrive/Documentos/Insper/Efootball/efootball/src/img/football_field.png')
        #template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        #template = cv2.resize(template, (320,320))/255
        #template = rgb_template_to_coord_conv_template(template)
        #self.template = template
    
    def calculate_metrics(self, frame, ball_mask, players_masks, ball_metrics):
        if len(ball_mask["boxes"]) != 0:
            ball_metrics.calculate_possession(frame, ball_mask, players_masks)
        
    def get_video_configs(self):
        cap = cv2.VideoCapture(self.args.path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_cap = cv2.VideoWriter(self.args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        return cap, n_frames, out_cap

    def process_frame(self, frame, out_cap):
        #if self.need_to_define_teams_colors:
        #    self.teams_kmeans.define_teams_colors(frame, self.person_detector)
        #    self.ball_metrics.init_possession(self.teams_kmeans.teams)
        #    self.need_to_define_teams_colors = False

        person_predictions = self.person_detector.detect_persons(frame)
        player_predictions = self.teams_segmentation.get_players(frame, person_predictions)
        draw_based_on_predictions(frame, player_predictions)
        #ball_predictions = self.ball_detector.detectBalls(frame)
        #frame_resized = cv2.resize(frame, (320,320))
        #keypoints_mask = self.keypoint_model(frame_resized)
        #players_info, balls_info = get_player_and_ball_informations(frame, person_predictions, ball_predictions, self.teams_kmeans)
        #draw_bboxes(frame, players_info, balls_info)
        #self.calculate_metrics(frame, ball_predictions, players_info, self.ball_metrics)
        #teams_number = self.teams_kmeans.teams
        #draw_metricis(teams_number=teams_number,
        #    team1_percentage = self.ball_metrics.get_possession_in_percentage(teams_number[0]),
        #    team2_percentage = self.ball_metrics.get_possession_in_percentage(teams_number[1]),
        #    frame=frame
        #)
        #players_points_homography = homography_player_estimation(keypoints_mask, players_info, (320,320))
        #draw_perspective_field(frame, self.template, players_points_homography)
        #if test:
        #    plt.imshow(frame)
        #    plt.show()
        #else:
        out_cap.write(frame)
    
    def process(self):
        cap, n_frames, out_cap = self.get_video_configs()
        progress_bar = tqdm.tqdm(total=n_frames)
        for i in range(50):
            _, frame = cap.read()
            if i == 0:
                self.teams_segmentation.define_teams_colors(frame)
            self.process_frame(frame, out_cap)
            progress_bar.update(1)
        progress_bar.close()
        cap.release()
        out_cap.release()
            