from utils import distance
import numpy as np


class player_ball_assigner:
    def __init__(self):
        self.min_dist = 70

    def ball_assign_on_frame(self, players, ball_bbox):
        x1, y1, x2, y2 = ball_bbox
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        assign_id = -1
        for track_id, player in players.items():
            min_dist = 7000
            player_bbox = player['bbox']
            left_dist = distance((cx, cy), (player_bbox[0], player_bbox[3]))
            right_dist = distance((cx, cy), (player_bbox[2], player_bbox[3]))

            dist = min(left_dist, right_dist)
            if dist < self.min_dist:
                if dist < min_dist:
                    min_dist = dist
                    assign_id = track_id

        return assign_id

    def annotate_player(self, track):

        for frame_num, players in enumerate(track['player']):
            ball_coor = track['ball'][frame_num][1]['bbox']
            assign_id = self.ball_assign_on_frame(players, ball_coor)
            if assign_id != -1:
                track['player'][frame_num][assign_id]['has_ball'] = True

        return track
