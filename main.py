import os
import cv2
from experiment.Camera_movement import CameraMovement
from src.Distance_And_Velocity import Measurement
from src.Team_Assigner import TeamAssigner
from src.Tracker import Tracker
from src.Minimap import MiniMap

input_video_path = "Personal_Project/Football_Analyst_Project/Inputs/Video_Project.mp4"
output_video_path = "Personal_Project/Football_Analyst_Project/Outputs/Match_Video_R.mp4"
stub_path = "Personal_Project/Football_Analyst_Project/Stubs/stub_minimap_ball_v3.pkl"
camera_stub_path = "Personal_Project/Football_Analyst_Project/Stubs/camera_movement_stub.pkl"

# 1. Initialize tracker & track objects
tracker = Tracker()
tracks = tracker.get_objects_tracked_from_video(
    input_video_path,
    batch_size=20,
    # Re-run detection so the temporal class stabilization is applied to fresh tracks.
    use_stub=True,
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

# 5. Project every frame onto the pitch once and store shared coordinates.
mini_map = MiniMap()
tracks = mini_map.add_pitch_positions_to_tracks(tracks)

# Use the source video's actual frame rate so elapsed time is measured correctly.
video_capture = cv2.VideoCapture(input_video_path)
frame_rate = video_capture.get(cv2.CAP_PROP_FPS) if video_capture.isOpened() else 0.0
video_capture.release()
if not frame_rate or frame_rate <= 0:
    frame_rate = 24.0

# 6. Calculate distance and velocity from those same pitch-meter coordinates.
measurement = Measurement(frame_rate=frame_rate, frame_window=12)
tracks = measurement.add_distance_and_velocity(tracks)

#compute ball possession
possession_info = tracker.calculate_ball_possession(
    tracks, frame_rate=frame_rate, max_distance=50.0, min_consecutive_frames=5
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
    minimap=mini_map,
)
print("thành công rồi chúc mừng đại ca")
