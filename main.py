import os
os.environ["OMP_NUM_THREADS"] = '1'
import argparse
import tensorflow as tf

from efootball.src.classes.models.detectron2_person.ModelPerson import PersonDetector
from efootball.src.classes.models.yolov5_ball.ModelBall import BallDetector as BallDetectorYolo
from efootball.src.classes.models.detectron2_ball.ModelBall import BallDetector
from efootball.src.classes.models.narya_field_homography.KeypointDetectorModel import KeypointDetectorModel
from efootball.src.classes.metrics.ball import BallMetrics
from efootball.src.classes.Processor import Processor
from efootball.src.classes.TeamsSegmentation import TeamsSegmentation

from efootball.src.constants.models import NARYA_KEYPOINT_DETECTOR, NARYA_WEIGTHS_NAME, NARYA_WEIGHTS_TOTAR

if __name__ == '__main__':
    print("Starting EFootBall...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to video', type=str, required=True)
    parser.add_argument('--out_video', help='path with detection results', type=str, required=True, default="./data/output")
    parser.add_argument('--test', help='test envrionment option', type=str, required=True, default="false")
    parser.add_argument('--player_threshold', help='player confidence detection threshold', type=float, default=0.6)
    parser.add_argument('--ball_threshold', help='ball confidence detection threshold', type=float, default=0.8)
    args = parser.parse_args()
    assert os.path.exists(args.path), 'Cannot open video: {}'.format(args.path)
    person_detector = PersonDetector(threshold=args.player_threshold)
    ball_detector = BallDetector(threshold=args.ball_threshold)
    ball_metrics = BallMetrics()
    kmeans = TeamsSegmentation(clusters=4)
    kp_model = KeypointDetectorModel(backbone='efficientnetb3', num_classes=29, input_shape=(320, 320))
    checkpoints = tf.keras.utils.get_file(NARYA_WEIGTHS_NAME, NARYA_KEYPOINT_DETECTOR, NARYA_WEIGHTS_TOTAR)
    kp_model.load_weights(checkpoints)
    processor = Processor(person_detector, ball_detector, kp_model, ball_metrics, kmeans, args)
    processor.process()