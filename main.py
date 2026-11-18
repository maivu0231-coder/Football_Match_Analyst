import os
import cv2
from Camera_movement import CameraMovement
from Distance_And_Velocity import Measurement
from Team_Assigner import TeamAssigner
from Tracker import Tracker
from View_Transform import ViewTransformer

input_video_path = "Personal_Project/Football_Analyst_Project/Inputs/Video_Input.mp4"
output_video_path = "Personal_Project/Football_Analyst_Project/Outputs/Match_Video_R.mp4"
stub_path = "Personal_Project/Football_Analyst_Project/Stubs/stub.pkl"
camera_stub_path = "Personal_Project/Football_Analyst_Project/Stubs/camera_movement_stub.pkl"

# 1. Initialize tracker & track objects
tracker = Tracker()
tracks = tracker.get_objects_tracked_from_video(
    input_video_path,
    batch_size=20,
    # Re-run detection so the temporal class stabilization is applied to fresh tracks.
    use_stub=False,
    stub_path=stub_path,
)

# 2. Assign team color
team_assigner = TeamAssigner()
tracks = team_assigner.assign_team_color_from_video(input_video_path, tracks)

# 3. Interpolate ball position & add positions
tracks = tracker.interpolation_ball(tracks)
tracks = tracker.add_position_to_track(tracks)

# 4. Compute camera movement and adjust player positions accordingly
camera_movement_tracker = CameraMovement()
camera_movements = camera_movement_tracker.get_camera_movement(
    input_video_path,
    use_stub=True,
    stub_path=camera_stub_path,
)
camera_movement_tracker.add_camera_position_to_tracks(tracks, camera_movements)

# 5. Using view transformation to convert pixel coordinates to meters
view_transformer = ViewTransformer()
tracks = view_transformer.add_transformed_position_to_tracks(tracks)

# 6. Add distance and velocity to tracks
measurement = Measurement(frame_rate=24, frame_window=5)
tracks = measurement.add_distance_and_velocity(tracks)

#compute ball possession
possession_info = tracker.calculate_ball_possession(
    tracks, frame_rate=24.0, max_distance=50.0, min_consecutive_frames=5
)
print(
    f"Kiểm soát bóng - Đội A: {possession_info['team_A']['possession_percentage']:.2f}% "
    f"({possession_info['team_A']['time_seconds']:.2f}s) | "
    f"Đội B: {possession_info['team_B']['possession_percentage']:.2f}% "
    f"({possession_info['team_B']['time_seconds']:.2f}s)"
)

# 8. Draw the annotated video with distance, velocity and ball possession information
tracker.annotate_and_save_video(
    input_video_path,
    output_video_path,
    tracks,
    measurement=measurement,
    possession_info=possession_info,
)
print("thành công rồi chúc mừng đại ca")
