import math

from efootball.src.utils.geometry import define_center_point
from efootball.src.utils.visualization import draw_line_between_points

class BallMetrics():
    def __init__(self):
        self.total_frames = 0
    
    def init_possession(self, teams):
        self.possession = {team: 0 for team in teams}
    
    def get_possession_in_percentage(self, team):
        if self.total_frames == 0:
            return 0
        possession = round((self.possession[team]/self.total_frames)*100, 2)
        return possession
    
    def get_metrics(self):
        metrics = dict()
        for team in self.possession:
            percentage = (self.possession[team]/self.total_frames)*100
            percentage = round(percentage, 2)
            metrics[team] = percentage
        return metrics

    def calculate_possession(self, frame, ball_mask, players_masks):
        ball_pos = ball_mask["boxes"][0]
        ball_center_position = define_center_point(ball_pos)
        closest_player = {"team": -1, "dist": math.inf, "position": [0,0]}
        for index in range(len(players_masks["boxes"])):
            player_position = players_masks["boxes"][index]
            player_team = players_masks["teams"][index]
            player_center_position = define_center_point(player_position)
            dist_for_ball = math.dist(ball_center_position, player_center_position)
            if dist_for_ball < closest_player["dist"]:
                closest_player["dist"] = dist_for_ball
                closest_player["team"] = player_team
                closest_player["position"] = player_center_position
        self.possession[closest_player["team"]] += 1
        self.total_frames += 1
        draw_line_between_points(frame, ball_center_position, closest_player["position"])