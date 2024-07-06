from ultralytics import YOLO
from detection import detection_cls
from utils import video_read
from utils import write_video
from camera_movement import camera_position_find
from view_transformation import viewTransformer
from find_distance_velocity import DistanceVelocityEstimator
import pickle


path_of_the_model = r'E:\python\basic_code\football_analysis_using_yolo\best.pt'
path_of_the_video = r'E:\python\basic_code\football_analysis_using_yolo\input_video.mp4'
file_path = r'E:\python\basic_code\football_analysis_using_yolo\info_dict.pkl'
yolo = YOLO(path_of_the_model)

model = detection_cls(yolo)
info = model.get_the_box_and_trancker_info(path_of_the_video)
model.add_position_to_track(info)
frames = video_read(path_of_the_video)
# find the position regarding the camera movement
cam_position = camera_position_find()
position = cam_position.camera_x_y(frames)
# add the position to the track dict
cam_position.adjust_camera_movment_position(info, position)
# display the cameras movement
frames = cam_position.display_cam_position(frames, position)
# transform the  camera position to original position in meter
view_transform = viewTransformer()
view_transform.add_transform_point(info)
# calculate the velocity and distance
dist_vel = DistanceVelocityEstimator()
dist_vel.calculateDistanceVelocity(info)
# plt the velocity and the total distance
frames = dist_vel.drawVelocityDistance(frames, info)
with open('info_dict.pkl', 'wb') as file:
    pickle.dump(info, file)
file.close()
# draw the ellips and tracker id to corresponding player and divide then according the team jersy color
out_frames = model.draw_elips_and_annotation(
    video_frames=frames, tracker_info=info)
# save the final output video
videoname = r'E:\python\basic_code\football_analysis_using_yolo\output_video.avi'
write_video(out_frames, videoname)
