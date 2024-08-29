import os
os.environ["OMP_NUM_THREADS"] = '1'
import argparse

from efootball.src.classes.Processor import Processor

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

    processor = Processor(args)
    processor.process()