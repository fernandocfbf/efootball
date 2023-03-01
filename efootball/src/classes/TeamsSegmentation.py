import numpy as np
from sklearn.cluster import KMeans

from efootball.src.utils.players import define_player_centroid

class TeamsSegmentation():
    def __init__(self, clusters):
        self.clusters = clusters
        self.teams = None
        self.kmeans = None
    
    def define_teams_colors(self, image, model):
        detections = model.detectPersons(image)
        mean_colors_for_players = list()
        for mask, label in zip(np.asarray(detections["masks"].to("cpu")), detections['labels']):
            if label == 0:
                _, color = define_player_centroid(image, mask)
                mean_colors_for_players.append(color)
        kmeans = KMeans(n_clusters = 4)
        kmeans.fit(mean_colors_for_players)
        most_common_color = [kmeans.predict([color]) for color in mean_colors_for_players]
        unique, counts = np.unique(most_common_color, return_counts=True)
        teams = unique[np.argsort(counts)[-2:]].tolist()
        self.kmeans = kmeans
        self.teams = teams
    
    def indentify_and_predict(self, color):
        cluster = self.kmeans.predict(color)
        is_player = cluster in self.teams
        return [is_player, int(cluster)]

