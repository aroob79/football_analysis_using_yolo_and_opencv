import cv2
import numpy as np
from utils import distance


class camera_position_find:
    def __init__(self):
        self.minimim_dist = 5

    def adjust_camera_movment_position(self, tracks, camera_position):
        for object1, object_list in tracks.items():
            for frame_num, track in enumerate(object_list):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    cam_position = camera_position[frame_num]
                    x_position_adj = -(cam_position[0]-position[0])
                    y_position_adj = -(cam_position[1]-position[1])
                    tracks[object1][frame_num][track_id]['adj_position'] = (
                        x_position_adj, y_position_adj)

    def camera_x_y(self, frames):
        gray_first_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        mask_feature = np.zeros_like(gray_first_frame)
        mask_feature[:100, :] = 1
        mask_feature[1000:, :] = 1
        old_feature = cv2.goodFeaturesToTrack(
            gray_first_frame, 150, 0.4, 3, blockSize=7, mask=mask_feature)
        camera_position = {}
        for i, frame in enumerate(frames):
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_feature, st, _ = cv2.calcOpticalFlowPyrLK(gray_first_frame, gray_frame, old_feature, None, winSize=(
                15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            # Select good points
            good_new = new_feature[st == 1]
            good_old = old_feature[st == 1]
            max_dist = 0
            camera_x, camera_y = 0, 0
            for _, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                dist = distance((a, b), (c, d))

                if dist > max_dist:
                    max_dist = dist
                    camera_x = c-a
                    camera_y = d-b
            if max_dist > self.minimim_dist:
                camera_position[i] = (camera_x, camera_y)
                old_feature = cv2.goodFeaturesToTrack(
                    gray_frame, 150, 0.4, 3, blockSize=7, mask=mask_feature)

            else:
                camera_position[i] = (0, 0)

            gray_first_frame = gray_frame.copy()

        return camera_position

    def display_cam_position(self, frames, camera_position):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            fram = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (600, 150), (255, 100, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, fram, 1-alpha, 0, fram)
            x, y = camera_position[frame_num]
            cv2.putText(fram, f'Camera_movement_x: {x:.2f}',
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(fram, f'Camera_movement_y: {y:.2f}',
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(fram)

        return np.array(output_frames)
