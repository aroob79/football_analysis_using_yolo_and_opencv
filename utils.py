import cv2
import numpy as np


def video_read(video_path):

    video = cv2.VideoCapture(video_path)
    frames = []

    while video.isOpened:
        ret, frame = video.read()
        if ret:
            frames.append(frame)
        else:
            break
    video.release()
    return frames


def write_video(frames, videoname):
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = 25

    resolution = (frames[0].shape[1], frames[0].shape[0])
    video = cv2.VideoWriter(videoname, codec, fps, resolution)
    for frame in frames:
        video.write(frame)
    video.release()
    return


def draw_elips(frame, bbox, color):

    x_cen = int((bbox[0]+bbox[2])/2.)
    y_cen = int(bbox[3])
    width = int(np.abs(bbox[0]-bbox[2]))
    frame = cv2.ellipse(frame, (x_cen, y_cen),
                        (width, int(0.5*width)), 0, -40, 240, color, 2)
    return frame


def draw_rect_and_put_tract_num(frame, bbox, color, tracker_id):
    x_cen = int((bbox[0]+bbox[2])//2.)
    upper_corner = (int(x_cen-10), int(bbox[3]-10))
    lower_corner = (int(x_cen+10), int(bbox[3]+10))
    frame = cv2.rectangle(frame, upper_corner, lower_corner, color, -1)
    text = f'{tracker_id}'
    position = (int(x_cen-5), int(bbox[3]+5))
    color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, text, position, font,
                        0.4, color, 1, cv2.LINE_AA)
    return frame


def draw_rect(frame, bbox, color):
    x_cen = int((bbox[0]+bbox[2])//2.)
    y_cen = int(bbox[1]-5.)
    rect_point = np.array([[x_cen, y_cen],
                           [x_cen-10, y_cen-10],
                           [x_cen+10, y_cen-10]])
    frame = cv2.drawContours(frame, [rect_point], 0, color, -1)
    frame = cv2.drawContours(frame, [rect_point], 0, (0, 0, 0), 2)
    return frame


def distance(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5


def find_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2


def find_foot_point(bbox):
    x1, _, x2, y2 = bbox
    return (x1+x2)/2, y2


if __name__ == '__main__':
    print(video_read(
        r'E:\python\basic_code\football_analysis_using_yolo\input_video.mp4'))
