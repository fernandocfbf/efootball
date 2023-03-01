import math
import cv2

from efootball.src.utils.geometry import define_center_point

class BallMetrics():
    def __init__(self):
        self.position = None
        self.total_frames = 0
        self.possession = None
    
    def init_possession(self, teams):
        self.possession = {team: 0 for team in teams}
    
    def get_possession_in_percentage(self, team):
        if self.total_frames == 0:
            return 0
        possession = round((self.possession[team]/self.total_frames)*100, 2)
        return possession

    def draw_line_to_closest_player(self, frame, ball_center_position, closest_player):
        cv2.line(frame, (ball_center_position[0], ball_center_position[1]),
            (closest_player["position"][0], closest_player["position"][1]),
            (0, 255, 0), thickness=2, lineType=4)
        
    def calculate_possession(self, frame, ball_mask, players_masks):
        ball_pos = ball_mask["boxes"][0]
        ball_center_position = define_center_point(ball_pos[0], ball_pos[2], ball_pos[1], ball_pos[3])
        closest_player = {"team": -1, "dist": math.inf, "position": [0,0]}
        for player_info in players_masks:
            poistion = player_info[0]
            player_center_position = define_center_point(poistion[2], poistion[3], poistion[0], poistion[1])
            dist_for_ball = math.dist(ball_center_position, player_center_position)
            if dist_for_ball < closest_player["dist"]:
                closest_player["dist"] = dist_for_ball
                closest_player["team"] = player_info[1]
                closest_player["position"] = player_center_position
        self.possession[closest_player["team"]] += 1
        self.total_frames += 1
        self.draw_line_to_closest_player(frame, ball_center_position, closest_player)