
from utils import video_read
from utils import draw_elips
from utils import draw_rect_and_put_tract_num
from utils import draw_rect
from utils import find_center
from utils import find_foot_point
from color_detection_and_team_assign import color_finder
from color_detection_and_team_assign import team_assigner
from player_ball_assign import player_ball_assigner
import supervision as sv
import pandas as pd
import pickle
import os


class detection_cls:
    def __init__(self, model):
        self.model = model
        self.tracker = sv.ByteTrack(track_activation_threshold=0.1, lost_track_buffer=3,
                                    minimum_matching_threshold=0.999999, frame_rate=5)
        self.color_find = color_finder()
        self.team_assin = team_assigner()
        self.player_assign = player_ball_assigner()

    def inter_polate_ball_position(self, track):
        ball_position = [list(val.values())[0]['bbox'] if len(
            list(val.values())) != 0 else [] for val in track['ball']]

        df = pd.DataFrame(ball_position, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        ball_position = [{1: {'bbox': val}} for val in df.to_numpy().tolist()]

        return ball_position

    def add_position_to_track(self, tracker):

        for object1, object_list in tracker.items():
            for frame_num, track in enumerate(object_list):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object1 == 'ball':
                        position = find_center(bbox)
                    else:
                        position = find_foot_point(bbox)

                    tracker[object1][frame_num][track_id]['position'] = position

    def get_the_box_and_trancker_info(self, video_path, exist_file_path=None):

        if exist_file_path is not None and os.path.exists(exist_file_path):
            with open(exist_file_path, 'rb') as file:
                info = pickle.load(file)
            file.close()
            return info
        # first get all the frame of the video
        frames = video_read(video_path)
        num_of_frame = len(frames)
        batch_num = 30
        detections = []
        for indx in range(0, num_of_frame, batch_num):
            detect = self.model.predict(
                frames[indx:indx+batch_num], save=False, conf=0.1)
            detections += detect
        num2_clss = detections[0].names
        clss2num = {k: v for v, k in num2_clss.items()}
        info = {'player': [],
                'ball': [],
                'referee': []}
        colors = []
        for frame_num, detection in enumerate(detections):
            detection = sv.Detections.from_ultralytics(detection)
            # convert goolkeeper to player
            for i, clss_ in enumerate(detection.class_id):
                if num2_clss[clss_] == 'goalkeeper':
                    detection.class_id[i] = clss2num['player']

            detections = self.tracker.update_with_detections(detection)
            info['player'].append({})
            info['ball'].append({})
            info['referee'].append({})

            for tracker_detection in detections:
                bbox = tracker_detection[0].tolist()
                cls_id = tracker_detection[3]
                tracker_id = tracker_detection[4]
                if cls_id == clss2num['player']:
                    color = self.color_find.get_color(frames[frame_num], bbox)
                    colors.append(color)
                    info['player'][frame_num][tracker_id] = {
                        'bbox': bbox, 'color': color, 'team': []}

                elif cls_id == clss2num['referee']:
                    info['referee'][frame_num][tracker_id] = {'bbox': bbox}

            for tracker_detection in detection:
                bbox = tracker_detection[0].tolist()
                cls_id = tracker_detection[3]
                if cls_id == clss2num['ball']:
                    info['ball'][frame_num][1] = {'bbox': bbox}
        # get the color for the team
        color1, color2 = self.team_assin.get_the_team_color(
            colors)

        for frame_num, track in enumerate(info['player']):
            for tracker_id, _ in track.items():
                color = info['player'][frame_num][tracker_id]['color']
                res = self.team_assin.get_the_team(color)
                info['player'][frame_num][tracker_id]['team'] = res+1

                if res == 0:
                    info['player'][frame_num][tracker_id]['color'] = color1
                else:
                    info['player'][frame_num][tracker_id]['color'] = color2

        # inter polate the ball position
        info['ball'] = self.inter_polate_ball_position(info)
        # assign the ball to nearest player
        info = self.player_assign.annotate_player(info)
        # save the file as pickle
        with open('info_dict.pkl', 'wb') as file:
            pickle.dump(info, file)
        file.close()

        return info

    def draw_elips_and_annotation(self, video_frames, tracker_info):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            out_frame = frame.copy()
            player_info = tracker_info['player'][frame_num]
            ball_info = tracker_info['ball'][frame_num]
            referees_info = tracker_info['referee'][frame_num]

            # draw the bounding box for player
            for tracker_id, bboxinfo in player_info.items():
                # first draw the elips
                color = bboxinfo.get('color', (0, 0, 255))
                out_frame = draw_elips(
                    out_frame, bboxinfo['bbox'], color)
                color = (255, 255, 255)
                out_frame = draw_rect_and_put_tract_num(
                    out_frame, bboxinfo['bbox'], color, tracker_id)
                if bboxinfo.get('has_ball', False):
                    out_frame = draw_rect(
                        out_frame, bboxinfo['bbox'], (0, 255, 255))

            for _, bboxinfo in referees_info.items():
                color = (0, 255, 255)
                out_frame = draw_elips(
                    out_frame, bboxinfo['bbox'], color)

            for _, bboxinfo in ball_info.items():
                color = (0, 255, 0)
                out_frame = draw_rect(
                    out_frame, bboxinfo['bbox'], color)

            output_frames.append(out_frame)

        return output_frames
