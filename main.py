import cv2
from Camera_movement import CameraMovement
from Distance_And_Velocity import Measurement
from Team_Assigner import TeamAssigner
from Tracker import Tracker
from View_Transform import ViewTransformer
input_video_path = "Personal_Project/Football_Analyst_Project/Inputs/Match_Video.mp4"
output_video_path = "Personal_Project/Football_Analyst_Project/Outputs/Match_Video_R.mp4"

# 1. Initialize tracker & track objects
tracker = Tracker()
tracks = tracker.get_objects_tracked_from_video(
    input_video_path,
    batch_size=20,
    use_stub=True,
    stub_path="Personal_Project/Football_Analyst_Project/Stubs/stub.pkl",
)

# 2. Assign team color
team_assigner = TeamAssigner()
tracks = team_assigner.assign_team_color_from_video(input_video_path, tracks)

# 3. Interpolate ball position & add positions
tracks = tracker.interpolation_ball(tracks)
tracks = tracker.add_position_to_track(tracks)

# compute camera movement and adjust player positions accordingly
camera_movement_tracker = CameraMovement()
camera_movements = camera_movement_tracker.get_camera_movement(
    input_video_path,
    use_stub=True,
    stub_path="Personal_Project/Football_Analyst_Project/Stubs/camera_movement_stub.pkl",
)
camera_movement_tracker.add_camera_position_to_tracks(tracks, camera_movements)
#using view transformation to convert pixel coordinates to meters
view_transformer = ViewTransformer()
tracks = view_transformer.add_transformed_position_to_tracks(tracks)
#  Add distance and velocity to tracks
measurement = Measurement(frame_rate=24, frame_window=5)
tracks = measurement.add_distance_and_velocity(tracks)

# draw the annotated video with distance and velocity information
tracker.annotate_and_save_video(
    input_video_path, output_video_path, tracks, measurement=measurement
)

print("Hoàn tất xử lý video!")