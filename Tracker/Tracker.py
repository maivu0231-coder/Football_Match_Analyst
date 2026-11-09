import os
import pickle
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
from Utils import get_center, get_foot, save_video
from Camera_movement import CameraMovement
from Distance_And_Velocity import Measurement
class Tracker:
    def __init__(self, model_path: str = "Personal_Project/Football_Analyst_Project/Models/last.pt",keypoint_model_path: str = "Personal_Project/Football_Analyst_Project/Models/yolov8x-pose.pt"):
        self.model = YOLO(model_path)
        self.keypoint_model = YOLO(keypoint_model_path)
        self.tracker = None

    def _init_tracker(self, fps: int = 24) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=0.5,#threshold for new track activation
            lost_track_buffer=fps,
            minimum_matching_threshold=0.8,
            frame_rate=fps,
            minimum_consecutive_frames=2
        )

    def get_objects_tracked_from_video(
        self, 
        video_path: str, 
        batch_size: int = 20, 
        use_stub: bool = False, 
        stub_path: str = "Personal_Project/Football_Analyst_Project/Stubs/stub.pkl"
    ) -> dict:
        if use_stub and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        self.tracker = self._init_tracker(fps=fps)

        tracks = {
            "player": [],
            "ball": [],
            "referee": [],
            "keypoints": []
        }

        frames_batch = []
        frame_idx = 0

        print(f"Bắt đầu tracking ({total_frames} frames)...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames_batch.append(frame)

            if len(frames_batch) == batch_size:
                results = self.model.predict(frames_batch, conf=0.15, verbose=False)
                keypoint_results = self.keypoint_model.predict(frames_batch, conf=0.15, verbose=False)
                #process keypoint results and add to tracks
                self._process_detections_to_tracks(results, keypoint_results, tracks, frame_idx)
                frame_idx += len(frames_batch)
                frames_batch = []

                if total_frames > 0:
                    print(f"\rTiến độ tracking: {(frame_idx / total_frames) * 100:.1f}% ({frame_idx}/{total_frames})", end="", flush=True)

        if len(frames_batch) > 0:
            results = self.model.predict(frames_batch, conf=0.15, verbose=False)
            keypoint_results = self.keypoint_model.predict(frames_batch, conf=0.15, verbose=False)
            self._process_detections_to_tracks(results, keypoint_results, tracks, frame_idx)
            frame_idx += len(frames_batch)

        print()
        cap.release()

        if use_stub:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def _process_detections_to_tracks(self, results,keypoint_results, tracks, start_idx)-> None:
        for offset, result in enumerate(results):
            curr_frame_idx = start_idx + offset
            cls_names = result.names

            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["referee"].append({})
            tracks["keypoints"].append({})
            # Extract keypoints using YOLO pose model
            if keypoint_results is not None and offset < len(keypoint_results):
                kp_result = keypoint_results[offset]
                if kp_result.keypoints is not None and len(kp_result.keypoints) > 0:
                    try:
                        keypoints = sv.KeyPoints.from_ultralytics(kp_result)
                        kp_id = 0
                        for p_idx in range(len(keypoints.xy)):
                            for k_idx in range(len(keypoints.xy[p_idx])):
                                pt = keypoints.xy[p_idx][k_idx]
                                conf = (
                                    float(keypoints.confidence[p_idx][k_idx])
                                    if keypoints.confidence is not None
                                    else 1.0
                                )
                                if conf > 0.3 and (pt[0] > 0 or pt[1] > 0):
                                    tracks["keypoints"][curr_frame_idx][kp_id] = {
                                        "position": [float(pt[0]), float(pt[1])],
                                        "confidence": conf,
                                    }
                                    kp_id += 1
                    except Exception:
                        pass

            detections = sv.Detections.from_ultralytics(result)
            if len(detections) == 0:
                self.tracker.update_with_detections(sv.Detections.empty())
                continue

            #ball detection
            ball_cls_ids = [k for k, v in cls_names.items() if v == "ball"]
            if ball_cls_ids:
                ball_id = ball_cls_ids[0]
                ball_mask = (detections.class_id == ball_id)
                
                if np.any(ball_mask):
                    ball_dets = detections[ball_mask]
                    # get ball with highest confidence
                    best_idx = np.argmax(ball_dets.confidence)
                    best_bbox = ball_dets.xyxy[best_idx].tolist()
                    best_conf = float(ball_dets.confidence[best_idx])

                    tracks["ball"][curr_frame_idx][1] = {
                        "bbox": best_bbox,
                        "confidence": best_conf,
                        "predicted": False
                    }
            #only track player and referee, not ball
            person_mask = np.isin(
                detections.class_id, 
                [k for k, v in cls_names.items() if v in ["player", "goalkeeper", "referee"]]
            )
            person_detections = detections[person_mask]

            if len(person_detections) > 0:
                tracked_persons = self.tracker.update_with_detections(person_detections)

                for xyxy, class_id, tracker_id in zip(
                    tracked_persons.xyxy,
                    tracked_persons.class_id,
                    tracked_persons.tracker_id
                ):
                    if tracker_id is None:
                        continue

                    cls_name = cls_names[int(class_id)]
                    bbox = xyxy.tolist()
                    t_id = int(tracker_id)

                    if cls_name in ["player", "goalkeeper"]:
                        tracks["player"][curr_frame_idx][t_id] = {
                            "bbox": bbox
                        }
                    elif cls_name == "referee":
                        tracks["referee"][curr_frame_idx][t_id] = {
                            "bbox": bbox
                        }
            else:
                self.tracker.update_with_detections(sv.Detections.empty())

    def draw_frame(self, frame: np.ndarray, frame_num: int, tracks: dict, camera_movement: list = None, camera_movement_tracker: CameraMovement = None) -> np.ndarray:
        #this function draws the annotations on a single frame 
        player_key = "players" if "players" in tracks else "player"
        player_tracks_list = tracks.get(player_key, [])

        player_tracks = player_tracks_list[frame_num] if frame_num < len(player_tracks_list) else {}
        ball_tracks = tracks.get("ball", [])[frame_num] if "ball" in tracks and frame_num < len(tracks["ball"]) else {}
        referee_tracks = tracks.get("referee", [])[frame_num] if "referee" in tracks and frame_num < len(tracks["referee"]) else {}
        keypoints_tracks = tracks.get("keypoints", [])[frame_num] if "keypoints" in tracks and frame_num < len(tracks["keypoints"]) else {}

        # draw player elipse
        for track_id, track in player_tracks.items():
            bbox = track["bbox"]
            assigned_color = track.get("team_color", (0, 0, 255))
            self.draw_elipse(frame, bbox, assigned_color, track_id)

        # draw ball indicator
        for track_id, track in ball_tracks.items():
            bbox = track["bbox"]
            self.draw_ball_indicator(frame, bbox, (0, 255, 255))

        #draw keypoints
        for track_id, track in keypoints_tracks.items():
            position = track["position"]
            self.draw_keypoint(frame, position, (0, 255, 0))

        # draw referee elipse
        for track_id, track in referee_tracks.items():
            bbox = track["bbox"]
            self.draw_elipse(frame, bbox, (255, 0, 0))

        # draw camera movement overlay if available
        if camera_movement is not None and camera_movement_tracker is not None:
            frame = camera_movement_tracker.draw_camera_movement(frame, camera_movement)

        return frame

    def draw_annotations(self, video_frames, tracks: dict, camera_movements: list = None):
        #if video_frames is a list, draw annotations on each frame and return the list
        #if video_frames is a generator, return a generator that yields annotated frames
        #if video_frames is a single frame (numpy array), draw annotations on it and return the frame
        camera_movement_tracker = CameraMovement() if camera_movements else None

        if isinstance(video_frames, list):
            for frame_num, frame in enumerate(video_frames):
                c_mov = camera_movements[frame_num] if camera_movements and frame_num < len(camera_movements) else None
                self.draw_frame(frame, frame_num, tracks, c_mov, camera_movement_tracker)
            return video_frames
        elif hasattr(video_frames, "__next__") or (hasattr(video_frames, "__iter__") and not isinstance(video_frames, np.ndarray)):
            def generator():
                for frame_num, frame in enumerate(video_frames):
                    c_mov = camera_movements[frame_num] if camera_movements and frame_num < len(camera_movements) else None
                    self.draw_frame(frame, frame_num, tracks, c_mov, camera_movement_tracker)
                    yield frame
            return generator()
        else:
            c_mov = camera_movements[0] if camera_movements else None
            return self.draw_frame(video_frames, 0, tracks, c_mov, camera_movement_tracker)

    def annotate_and_save_video(self, input_video_path: str, output_video_path: str, tracks: dict, measurement=None) -> None:

        camera_movement_tracker = CameraMovement()
        camera_movements = camera_movement_tracker.get_camera_movement(
            input_video_path,
            use_stub=True, 
            stub_path="Personal_Project/Football_Analyst_Project/Stubs/camera_movement_stub.pkl"
        )
        camera_movement_tracker.add_camera_position_to_tracks(tracks, camera_movements)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Khong the mo: {input_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        def frame_stream():
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                c_mov = camera_movements[frame_num] if frame_num < len(camera_movements) else None
                frame = self.draw_frame(frame, frame_num, tracks, c_mov, camera_movement_tracker)

                # draw distance and velocity if measurement is provided
                if measurement is not None:
                    frame = measurement.draw_distance_and_velocity(frame, tracks, frame_num)

                yield frame
                frame_num += 1

            cap.release()

        # save video using the frame stream generator
        save_video(frame_stream(), output_video_path, fps=fps)
    def add_position_to_track(self, tracks: dict) -> dict:
        for obj in tracks:
            if obj == "keypoints":
                continue
            for frame_num, frame in enumerate(tracks[obj]):
                for track_id, track in frame.items():
                    if "bbox" not in track:
                        continue
                    bbox = track["bbox"]
                    if obj == "ball":
                        foot = get_foot(bbox)
                        tracks[obj][frame_num][track_id]["position"] = foot
                    else:
                        center = get_foot(bbox)
                        tracks[obj][frame_num][track_id]["position"] = center
        return tracks

    def interpolation_ball(self, tracks: dict) -> dict:
        ball_positions = [x.get(1, {}).get("bbox", []) for x in tracks["ball"]]
        ball_df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        ball_df_interpolated = ball_df.interpolate().bfill()
        tracks["ball"] = [{1: {"bbox": row.values.tolist()}} for _, row in ball_df_interpolated.iterrows()]
        return tracks

    def draw_keypoint(self, frame: any, position: list, color: any) -> any:
        x, y = [int(p) for p in position]
        cv2.circle(frame, (x, y), 5, color, -1)
        return frame

    def draw_elipse(self, frame: any, bbox: list, color: any, track_id: int = None) -> any:
        x1, y1, x2, y2 = [int(x) for x in bbox]
        center_x = int((x1 + x2) / 2)
        center_y = y2
        center = (center_x, center_y)

        width = x2 - x1
        axes = (int(width / 2), int(width / 6))

        angle = 0
        start_angle = -45 
        end_angle = 225  

        cv2.ellipse(
            frame, center, axes, angle, start_angle, end_angle, color, thickness=2
        )

        if track_id is not None:
            text = str(track_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness=1
            )
            text_x = center_x - int(text_width / 2)
            text_y = center_y + int(axes[1]) + 15

            cv2.putText(
                frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness=2
            )  
            cv2.putText(
                frame, text, (text_x, text_y), font, font_scale, color, thickness=1
            )  

        return frame

    def draw_ball_indicator(self, frame: any, bbox: list, color: any) -> any:
        x1, y1, x2, y2 = [int(x) for x in bbox]
        center_x = int((x1 + x2) / 2)
        top_y = y1 - 5 
        triangle_width = 16 
        triangle_height = 12 

        half_w = int(triangle_width / 2)
        pt1 = (center_x, top_y) 
        pt2 = (center_x - half_w, top_y - triangle_height)  
        pt3 = (center_x + half_w, top_y - triangle_height)  
        pts = np.array([pt1, pt2, pt3], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], color)

        return frame