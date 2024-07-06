import cv2
import numpy as np


class viewTransformer:
    def __init__(self):
        self.actual_width = 68
        self.actual_length = 29.1666

        self.actual_width2 = 68
        self.actual_length2 = 37.92

        self.pixel_of_vertex1 = np.array([[260, 258],
                                         [1050, 245],
                                         [100, 1030],
                                         [1930, 880]], dtype=np.float32)
        self.pixel_of_vertex2 = np.array([[700, 290],
                                          [10, 810],
                                          [1450, 300],
                                          [1850, 880]], dtype=np.float32)
        self.target_vertex1 = np.array([[0, 0],
                                       [self.actual_length, 0],
                                       [0, self.actual_width],
                                       [self.actual_length, self.actual_width]], dtype=np.float32)

        self.target_vertex2 = np.array([[0, 0],
                                       [self.actual_length2, 0],
                                       [0, self.actual_width2],
                                       [self.actual_length2, self.actual_width2]], dtype=np.float32)

        self.transform1 = cv2.getPerspectiveTransform(
            self.pixel_of_vertex1, self.target_vertex1)

        self.transform2 = cv2.getPerspectiveTransform(
            self.pixel_of_vertex2, self.target_vertex2)

    def transformPoint(self, point):
        x, y = point
        is_inside1 = cv2.pointPolygonTest(self.pixel_of_vertex1, (x, y), False)
        is_inside2 = cv2.pointPolygonTest(self.pixel_of_vertex2, (x, y), False)

        point = np.array([[[x, y]]], dtype=np.float32)
        if is_inside1 > 0:
            transform_point = cv2.perspectiveTransform(point, self.transform1)
            return transform_point.reshape((-1, 2))
        elif is_inside2 > 0:
            transform_point = cv2.perspectiveTransform(point, self.transform2)
            return transform_point.ravel()

        else:
            return None

    def add_transform_point(self, tracks):
        for object1, object_list in tracks.items():
            for frame_num, track in enumerate(object_list):
                for track_id, track_info in track.items():
                    position = track_info['adj_position']
                    transformed_point = self.transformPoint(position)
                    if transformed_point is not None:
                        transformed_point = transformed_point.tolist()

                    tracks[object1][frame_num][track_id]['transformed_point'] = transformed_point
