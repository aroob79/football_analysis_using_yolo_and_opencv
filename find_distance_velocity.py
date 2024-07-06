from utils import distance
from utils import find_foot_point
import numpy as np 
import cv2


class DistanceVelocityEstimator:
    def __init__(self):
        self.window_size = 3
        self.frame_rate = 25.0

    def calculateDistanceVelocity(self, trackes):
        player_track = trackes['player']
        total_distance = {}
        total_number_of_frames = len(player_track)
        for frame_num in range(0, total_number_of_frames):
            last_frame_num = min(frame_num+self.window_size,
                                 total_number_of_frames-1)
            thresh = total_number_of_frames - self.window_size

            for track_id, track in player_track[frame_num].items():
                if track_id in player_track[last_frame_num]:
                    start_position = track['transformed_point']
                    last_position = player_track[last_frame_num][track_id]['transformed_point']

                    if track_id not in total_distance:
                        total_distance[track_id] = 0

                    if (start_position is not None) and (last_position is not None):
                        start_position=np.array(start_position).ravel()
                        last_position=np.array(last_position).ravel()
                        distance_covert = distance(
                            start_position, last_position)
                        if frame_num > thresh:
                            time_duration = (
                                frame_num-thresh)/self.frame_rate
                        else:
                            time_duration = (
                                last_frame_num-frame_num)/self.frame_rate

                        velocity = distance_covert/time_duration

                        total_distance[track_id] += distance_covert

                    trackes['player'][frame_num][track_id]['total_distance'] = total_distance[track_id]
                    trackes['player'][frame_num][track_id]['velocity'] = velocity

    def drawVelocityDistance(self, frames, trackes):
        output_frames = []
        player_track = trackes['player']
        for frame_num, info in enumerate(player_track):
            frame = frames[frame_num]
            for _, box_info in info.items():
                velocity = box_info.get('velocity', None)
                total_distance = box_info.get('total_distance', None)
                if velocity is not None and total_distance is not None:
                    bbox = box_info.get('bbox', 0)
                    get_foot_position = find_foot_point(bbox)
                    x, y = int(get_foot_position[0]), int(get_foot_position[1])
                    cv2.putText(
                        frame, f'{velocity :.2f} m/s', (x, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    cv2.putText(
                        frame, f'{total_distance :.2f} m', (x, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames
