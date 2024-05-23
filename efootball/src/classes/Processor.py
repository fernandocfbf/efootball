import cv2
import tqdm

from efootball.src.classes.models.TeamsSegmentation import TeamsSegmentation
from efootball.src.classes.metrics.BallMetrics import BallMetrics
from efootball.src.classes.models.PersonDetector import PersonDetector
from efootball.src.classes.models.BallDetector import BallDetector

from efootball.src.utils.geometry import define_center_point
from efootball.src.utils.visualization import draw_based_on_predictions, draw_circle, draw_metricis

from efootball.src.constants.teams import BALL_COLOR

class Processor():
    def __init__(self, args):
        self.args = args

        #models
        self.person_detector = PersonDetector(threshold=0.6)
        self.teams_segmentation = TeamsSegmentation(num_clusters=4)
        self.ball_detector = BallDetector(threshold=0.8)
        
        #metrics
        self.ball_metrics = BallMetrics()
        
        #kp_model = KeypointDetectorModel(backbone='efficientnetb3', num_classes=29, input_shape=(320, 320))
        #checkpoints = tf.keras.utils.get_file(NARYA_WEIGTHS_NAME, NARYA_KEYPOINT_DETECTOR, NARYA_WEIGHTS_TOTAR)
        #kp_model.load_weights(checkpoints)
        #self.keypoint_model = kp_model
        #template = cv2.imread('C:/Users/ferna/OneDrive/Documentos/Insper/Efootball/efootball/src/img/football_field.png')
        #template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        #template = cv2.resize(template, (320,320))/255
        #template = rgb_template_to_coord_conv_template(template)
        #self.template = template
    
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
        
        if len(ball_predictions["boxes"]) > 0:
            self.ball_metrics.calculate_possession(frame, ball_predictions, player_predictions)
            ball_position = define_center_point(ball_predictions["boxes"][0])
            draw_circle(frame, ball_position, BALL_COLOR)

        draw_based_on_predictions(frame, player_predictions)
        draw_metricis(frame, self.ball_metrics.get_metrics())

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
                self.ball_metrics.init_possession(self.teams_segmentation.teams)
            self.process_frame(frame, out_cap)
            progress_bar.update(1)
        progress_bar.close()
        cap.release()
        out_cap.release()
            