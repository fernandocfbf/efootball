import numpy as np
import cv2
from sklearn.cluster import KMeans

from efootball.src.classes.models.PersonDetector import PersonDetector

class TeamsSegmentation():
    def __init__(self, num_clusters):
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.person_detector = PersonDetector(threshold=0.6)

    def define_mean_color(self, array):
        transpose_matriz = array.T
        red = np.mean(transpose_matriz[0])
        blue = np.mean(transpose_matriz[1])
        green = np.mean(transpose_matriz[2])
        return [red, blue, green]
    
    def get_player_avg_color(self, frame, item_mask):
        kernel = (item_mask * 255).astype('uint8')
        result = cv2.bitwise_and(frame, frame, mask=kernel)
        result_reshape = result.reshape((result.shape[0] * result.shape[1], 3))
        only_player_pixels = result_reshape[result_reshape.sum(axis=(1)) != 0]
        player_avg_color = self.define_mean_color(only_player_pixels)
        return player_avg_color

    def define_teams_colors(self, image):
        detections = self.person_detector.detect_persons(image)
        mean_colors_for_players = list()
        for mask, label in zip(np.asarray(detections["masks"].to("cpu")), detections['labels']):
            if label == 0:
                color = self.get_player_avg_color(image, mask)
                mean_colors_for_players.append(color)
        self.kmeans.fit(mean_colors_for_players)
        most_common_color = [self.kmeans.predict([color]) for color in mean_colors_for_players]
        unique, counts = np.unique(most_common_color, return_counts=True)
        teams = unique[np.argsort(counts)[-2:]].tolist()
        self.teams = teams
    
    def get_players(self, frame, person_predictions):
        players = {"masks":[], "teams":[]}
        for mask in np.asarray(person_predictions["masks"].to("cpu")):
            player_avg_color = self.get_player_avg_color(frame, mask)
            player_team = self.kmeans.predict([player_avg_color])[0]
            if player_team in self.teams:
                players["masks"].append(mask)
                players["teams"].append(player_team)
        return players
