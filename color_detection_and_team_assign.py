from sklearn.cluster import KMeans
import numpy as np


class color_finder:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)

    def get_color(self, frame, bbox):
        xmin, ymin, xmax, ymax = bbox
        player_pic = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        # take the top half of the image
        h, w, _ = player_pic.shape
        top_half_pic = player_pic[:h//2, :]
        h, w, _ = top_half_pic.shape
        top_half_pic = top_half_pic.reshape((-1, 3))
        result = self.kmeans.fit_predict(top_half_pic)
        result = result.reshape((h, w))

        # find the class of the background
        value, count = np.unique(
            (result[0, 0], result[0, -1], result[-5, 0], result[-5, -1]), return_counts=True)
        bg_label = value[np.argmax(count)]

        # foreground label and color
        fg_label = 1-bg_label
        color = self.kmeans.cluster_centers_[fg_label]

        return color


class team_assigner:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)

    def get_the_team_color(self, colors):
        self.kmeans.fit(colors)
        team_color1 = self.kmeans.cluster_centers_[0]
        team_color2 = self.kmeans.cluster_centers_[1]

        return team_color1, team_color2

    def get_the_team(self, color):
        if color.ndim == 1:
            color = color[np.newaxis, :]

        result = self.kmeans.predict(color)
        return result
