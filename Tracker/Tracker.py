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
from inference import get_model

class Tracker:
    def __init__(
        self,
        model_path: str = "Personal_Project/Football_Analyst_Project/Models/last.pt",
        keypoint_model_path="Personal_Project/Football_Analyst_Project/Models/kp.pt",
        api_key=None
    ):
        #model for player,ball
        self.model = YOLO(model_path)

        self.keypoint_model =YOLO(keypoint_model_path) if keypoint_model_path else None
        self.tracker = None
        # Stateful ball filtering is kept across inference batches.
        self._previous_ball_position = None
        self._previous_ball_frame = None
        self._previous_gray_frame = None
        self._ball_static_frames = 0
        self._ball_last_motion_position = None

    @staticmethod
    def _ball_center(box):
        x1, y1, x2, y2 = map(float, box)
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    @staticmethod
    def _ball_box_is_valid(box, frame_shape):
        x1, y1, x2, y2 = map(float, box)
        width, height = x2 - x1, y2 - y1
        frame_h, frame_w = frame_shape[:2]
        area = width * height
        ratio = width / height if height > 0 else float("inf")
        # A ball is compact and close to round. The previous 0.25% area limit
        # admitted boxes thousands of pixels wide on broadcast footage.
        return (
            width >= 2
            and height >= 2
            and area <= frame_w * frame_h * 0.0008
            and 0.55 <= ratio <= 1.8
        )

    @staticmethod
    def _ball_motion_score(box, previous_gray, current_gray):
        """Count changed pixels in the candidate ROI; scores are in pixel units."""
        x1, y1, x2, y2 = map(int, box)
        h, w = current_gray.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0
        before = previous_gray[y1:y2, x1:x2]
        after = current_gray[y1:y2, x1:x2]
        delta = cv2.absdiff(before, after)
        # Ignore small compression/lighting changes so the >=7 score reflects
        # meaningful motion in the tiny ball-sized ROI.
        return int(cv2.countNonZero(cv2.threshold(delta, 24, 255, cv2.THRESH_BINARY)[1]))

    def _init_tracker(self, fps: int = 24) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=0.5,  # threshold for new track activation
            lost_track_buffer=fps,
            minimum_matching_threshold=0.8,
            frame_rate=fps,
            minimum_consecutive_frames=2,
        )



    def _predict_keypoints(self, frames: list, box_conf: float = 0.5, kp_conf: float = 0.5):
        if self.keypoint_model is None or not frames:
            return None

        try:
            # Case 1: YOLO Inference
            if isinstance(self.keypoint_model, YOLO):
                # Lần lọc 1: Lọc Bounding Box sân bóng (conf=box_conf) và chỉ lấy 1 sân (max_det=1)
                results = self.keypoint_model.predict(
                    frames,
                    conf=box_conf,
                    max_det=1,
                    imgsz=960,
                    verbose=False
                )

                # Lần lọc 2: Lọc từng keypoint có độ tự tin < kp_conf
                for res in results:
                    if res.keypoints is not None and res.keypoints.data is not None and len(res.keypoints.data) > 0:
                        # res.keypoints.data có shape (num_boxes, num_keypoints, 3) -> [x, y, conf]
                        kp_data = res.keypoints.data.clone()
                        low_conf_mask = kp_data[..., 2] < kp_conf
                        kp_data[low_conf_mask] = 0.0  # Đưa x, y, conf về 0 nhưng giữ nguyên index điểm
                        res.keypoints.data = kp_data

                return results

            # Case 2: Roboflow Inference
            elif hasattr(self.keypoint_model, "infer"):
                results = []

                for frame in frames:
                    # Lần lọc 1: Lọc Bounding Box từ API/SDK của Roboflow
                    result = self.keypoint_model.infer(
                        frame,
                        confidence=box_conf,
                        imgsz=960
                    )[0]

                    # Lần lọc 2: Lọc từng keypoint trong danh sách predictions
                    if hasattr(result, "predictions") and result.predictions:
                        # Giữ lại prediction có độ tự tin cao nhất (tương đương max_det=1)
                        result.predictions = [max(result.predictions, key=lambda p: p.confidence)]
                        
                        for kp in result.predictions[0].keypoints:
                            if kp.confidence < kp_conf:
                                kp.x = 0.0
                                kp.y = 0.0
                                kp.confidence = 0.0

                    results.append(result)

                return results

            else:
                raise TypeError("Unsupported keypoint model")

        except Exception as e:
            print(f"Keypoint prediction error: {e}")
            return None

    def get_objects_tracked_from_video(
        self,
        video_path: str="Personal_Project/Football_Analyst_Project/Inputs/Video_Project.mp4",
        batch_size: int = 20,
        use_stub: bool = False,
        stub_path: str = "Personal_Project/Football_Analyst_Project/Stubs/stub.pkl",
    ) -> dict:

        video_path = video_path
        stub_path = stub_path

        if use_stub and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            self.tracker = self._init_tracker(fps=fps)

            tracks = {
                "player": [],
                "ball": [],
                "referee": [],
                "keypoints": [],
            }

            frames_batch = []
            frame_idx = 0

            print(f"Bắt đầu tracking ({total_frames} frames)...")

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frames_batch.append(frame)

                if len(frames_batch) >= batch_size:

                    # Object detection
                    results = self.model.predict(
                        frames_batch,
                        conf=0.25,
                        verbose=False,
                        imgsz=1280

                    )

                    # Field keypoint detection
                    keypoint_results = self._predict_keypoints(
                        frames_batch
                    )

                    # Process
                    self._process_detections_to_tracks(
                        results,
                        keypoint_results,
                        tracks,
                        frame_idx,
                        frames_batch,
                    )

                    frame_idx += len(frames_batch)
                    frames_batch = []

                    if total_frames > 0:
                        progress = frame_idx / total_frames * 100

                        print(
                            f"\rTiến độ tracking: {progress:.1f}% "
                            f"({frame_idx}/{total_frames})",
                            end="",
                            flush=True
                        )

            #process remaining frames if any
            if frames_batch:

                results = self.model.predict(
                    frames_batch,
                    conf=0.25,
                    verbose=False,
                    imgsz=1280


                )

                keypoint_results = self._predict_keypoints(
                    frames_batch
                )

                self._process_detections_to_tracks(
                    results,
                    keypoint_results,
                    tracks,
                    frame_idx,
                    frames_batch,
                )

                frame_idx += len(frames_batch)

            # Stabilize class over the full lifetime of each ByteTrack ID.
            # This can correct longer 1-2 second label flips as well as single
            # frame flicker, provided the ID mostly belongs to one person.
            self._stabilize_person_classes(tracks)

            print()

            # save stub if requested
            if use_stub:
                os.makedirs(os.path.dirname(stub_path), exist_ok=True)

                with open(stub_path, "wb") as f:
                    pickle.dump(tracks, f)

            return tracks

        finally:
            cap.release()

    @staticmethod
    def _stabilize_person_classes(tracks: dict) -> None:
        """Use each track ID's majority label over its full lifetime."""
        player_frames = tracks.get("player", [])
        referee_frames = tracks.get("referee", [])
        observations = {}

        for label, frames in (("player", player_frames), ("referee", referee_frames)):
            for frame_idx, frame_tracks in enumerate(frames):
                for track_id in frame_tracks:
                    observations.setdefault(track_id, []).append((frame_idx, label))

        smoothed_labels = {}
        for track_id, items in observations.items():
            player_votes = sum(label == "player" for _, label in items)
            referee_votes = sum(label == "referee" for _, label in items)
            if player_votes == referee_votes:
                continue
            stable_label = "player" if player_votes > referee_votes else "referee"
            for frame_idx, _ in items:
                smoothed_labels[(frame_idx, track_id)] = stable_label

        for (frame_idx, track_id), target_label in smoothed_labels.items():
            source_label = "referee" if target_label == "player" else "player"
            source_frames = tracks.get(source_label, [])
            target_frames = tracks.get(target_label, [])
            if frame_idx >= len(source_frames) or frame_idx >= len(target_frames):
                continue
            track_data = source_frames[frame_idx].pop(track_id, None)
            if track_data is not None:
                target_frames[frame_idx][track_id] = track_data

        
    def _process_detections_to_tracks(
        self, results, keypoint_results, tracks, start_idx, frames=None
    ) -> None:

        for offset, result in enumerate(results):

            curr_frame_idx = start_idx + offset
            cls_names = result.names

            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["referee"].append({})
            tracks["keypoints"].append({})
            #extract keypoints if available
            if keypoint_results is not None and offset < len(keypoint_results):

                kp_result = keypoint_results[offset]

                try:
                    # Ultralytics pose results expose .keypoints.xy/.conf.
                    # Roboflow inference results need Supervision conversion.
                    if hasattr(kp_result, "keypoints") and kp_result.keypoints is not None:
                        kp_xy = kp_result.keypoints.xy.cpu().numpy()
                        kp_conf_obj = kp_result.keypoints.conf
                        kp_conf = (
                            kp_conf_obj.cpu().numpy()
                            if kp_conf_obj is not None
                            else None
                        )
                    else:
                        kp_data = sv.KeyPoints.from_inference(kp_result)
                        kp_xy = kp_data.xy
                        kp_conf = kp_data.confidence

                    if kp_xy is not None and len(kp_xy) > 0:
                        next_id = 0
                        for p_idx in range(len(kp_xy)):
                            for k_idx in range(len(kp_xy[p_idx])):
                                pt = kp_xy[p_idx][k_idx]
                                conf = (
                                    float(kp_conf[p_idx][k_idx])
                                    if kp_conf is not None
                                    else 1.0
                                )
                                if conf <= 0.3 or not np.isfinite(pt).all():
                                    continue

                                tracks["keypoints"][curr_frame_idx][next_id] = {
                                    "position": [float(pt[0]), float(pt[1])],
                                    "confidence": conf,
                                }
                                next_id += 1

                except Exception as e:
                    print(
                        f"Keypoint processing error at frame "
                        f"{curr_frame_idx}: {e}"
                    )
            #convert ultralytics result to supervision Detections
            detections = sv.Detections.from_ultralytics(result)

            if len(detections) == 0:
                self.tracker.update_with_detections(
                    sv.Detections.empty()
                )
                continue
            #ball detection
            ball_cls_ids = [
                k for k, v in cls_names.items() if v == "ball"
            ]

            current_gray = None
            if frames is not None and offset < len(frames):
                current_gray = cv2.cvtColor(frames[offset], cv2.COLOR_BGR2GRAY)

            if ball_cls_ids:

                ball_id = ball_cls_ids[0]
                ball_mask = detections.class_id == ball_id

                if np.any(ball_mask):

                    ball_dets = detections[ball_mask]
                    valid = []
                    for bbox, confidence in zip(ball_dets.xyxy, ball_dets.confidence):
                        bbox = bbox.astype(float)
                        confidence = float(confidence)
                        if not self._ball_box_is_valid(bbox, result.orig_shape):
                            continue
                        # Frame difference motion gate (>= 7 changed pixels).
                        if current_gray is not None and self._previous_gray_frame is not None:
                            motion_score = self._ball_motion_score(bbox, self._previous_gray_frame, current_gray)
                            # Keep same-location candidates long enough to detect and
                            # clear a stationary false positive after seven frames.
                            stationary_candidate = (
                                self._previous_ball_position is not None
                                and np.linalg.norm(self._ball_center(bbox) - self._previous_ball_position) < 1.0
                            )
                            if motion_score < 7 and not stationary_candidate:
                                continue
                        valid.append((bbox.tolist(), confidence, self._ball_center(bbox)))

                    if valid:
                        if self._previous_ball_position is None:
                            # First observation: confidence gate and highest confidence.
                            eligible = [item for item in valid if item[1] >= 0.35]
                            selected = max(eligible, key=lambda item: item[1]) if eligible else None
                        else:
                            # Minimize distance - 40 * confidence from prior position.
                            selected = min(
                                valid,
                                key=lambda item: float(np.linalg.norm(item[2] - self._previous_ball_position)) - 40.0 * item[1],
                            )

                        if selected is not None:
                            best_bbox, best_conf, position = selected
                            if self._ball_last_motion_position is not None and np.linalg.norm(position - self._ball_last_motion_position) < 1.0:
                                self._ball_static_frames += 1
                            else:
                                self._ball_static_frames = 0
                                self._ball_last_motion_position = position.copy()

                            if self._ball_static_frames >= 7:
                                # Treat a persistent stationary detection as a false positive.
                                self._previous_ball_position = None
                                self._previous_ball_frame = None
                                self._ball_last_motion_position = None
                                self._ball_static_frames = 0
                            else:
                                self._previous_ball_position = position
                                self._previous_ball_frame = curr_frame_idx
                                tracks["ball"][curr_frame_idx][1] = {
                                    "bbox": best_bbox,
                                    "confidence": best_conf,
                                    "predicted": False,
                                }

            if current_gray is not None:
                self._previous_gray_frame = current_gray
            #player and referee detection
            person_mask = np.isin(
                detections.class_id,
                [
                    k
                    for k, v in cls_names.items()
                    if v in ["player", "goalkeeper", "referee"]
                ],
            )

            person_detections = detections[person_mask]

            if len(person_detections) > 0:

                tracked_persons = self.tracker.update_with_detections(
                    person_detections
                )

                for xyxy, class_id, tracker_id in zip(
                    tracked_persons.xyxy,
                    tracked_persons.class_id,
                    tracked_persons.tracker_id,
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
                self.tracker.update_with_detections(
                    sv.Detections.empty()
                )

    def calculate_ball_possession(
        self,
        tracks: dict,
        frame_rate: float = 24.0,
        max_distance: float = 50.0,
        min_consecutive_frames: int = 5,
        control_level: str = "player",
    ) -> dict:
        player_key = "players" if "players" in tracks else "player"
        player_tracks_list = tracks.get(player_key, [])
        ball_tracks_list = tracks.get("ball", [])

        total_frames = max(len(player_tracks_list), len(ball_tracks_list))
        if total_frames == 0:
            #return empty possession info if no frames are available
            return {
                "team_A": {
                    "team_id": 1,
                    "team_name": "Team A",
                    "frames": 0,
                    "time_seconds": 0.0,
                    "possession_ratio": 0.0,
                    "possession_percentage": 0.0,
                },
                "team_B": {
                    "team_id": 2,
                    "team_name": "Team B",
                    "frames": 0,
                    "time_seconds": 0.0,
                    "possession_ratio": 0.0,
                    "possession_percentage": 0.0,
                },
                1: {
                    "team_id": 1,
                    "team_name": "Team A",
                    "frames": 0,
                    "time_seconds": 0.0,
                    "possession_ratio": 0.0,
                    "possession_percentage": 0.0,
                },
                2: {
                    "team_id": 2,
                    "team_name": "Team B",
                    "frames": 0,
                    "time_seconds": 0.0,
                    "possession_ratio": 0.0,
                    "possession_percentage": 0.0,
                },
                "frame_rate": frame_rate,
                "max_distance": max_distance,
                "min_consecutive_frames": min_consecutive_frames,
                "team_control_per_frame": [],
                "player_control_per_frame": [],
                "cumulative_possession": [],
            }

        raw_player_control = [None] * total_frames
        raw_team_control = [None] * total_frames

        #compute the distance between the ball and each player for each frame, and assign control to the closest player within max_distance
        for f_idx in range(total_frames):
            ball_dict = (
                ball_tracks_list[f_idx] if f_idx < len(ball_tracks_list) else {}
            )
            player_dict = (
                player_tracks_list[f_idx] if f_idx < len(player_tracks_list) else {}
            )

            if not ball_dict or not player_dict:
                continue

            ball_info = (
                ball_dict.get(1, next(iter(ball_dict.values())))
                if ball_dict
                else None
            )
            if not ball_info or "bbox" not in ball_info:
                continue

            ball_bbox = ball_info["bbox"]
            if "position" in ball_info:
                ball_pos = np.array(ball_info["position"], dtype=np.float32)
            else:
                ball_pos = np.array(
                    [
                        (ball_bbox[0] + ball_bbox[2]) / 2.0,
                        (ball_bbox[1] + ball_bbox[3]) / 2.0,
                    ],
                    dtype=np.float32,
                )

            min_dist = float("inf")
            best_player_id = None
            best_team_id = None

            for p_id, p_info in player_dict.items():
                if "bbox" not in p_info:
                    continue
                p_bbox = p_info["bbox"]

                # compute the distance from the ball to the player's foot (bottom center of the bbox)
                foot_center = np.array(
                    [(p_bbox[0] + p_bbox[2]) / 2.0, p_bbox[3]], dtype=np.float32
                )
                foot_left = np.array([p_bbox[0], p_bbox[3]], dtype=np.float32)
                foot_right = np.array([p_bbox[2], p_bbox[3]], dtype=np.float32)

                d_center = np.linalg.norm(foot_center - ball_pos)
                d_left = np.linalg.norm(foot_left - ball_pos)
                d_right = np.linalg.norm(foot_right - ball_pos)
                d_foot = min(d_center, d_left, d_right)

                if "position" in p_info:
                    d_pos = np.linalg.norm(
                        np.array(p_info["position"], dtype=np.float32) - ball_pos
                    )
                    d_foot = min(d_foot, d_pos)

                if d_foot < min_dist:
                    min_dist = d_foot
                    best_player_id = p_id
                    best_team_id = p_info.get("team")

            # assign control to the closest player if within max_distance(50px)
            if min_dist <= max_distance and best_player_id is not None:
                raw_player_control[f_idx] = best_player_id
                raw_team_control[f_idx] = best_team_id

        # keep for 5 frames
        confirmed_team_control = [None] * total_frames
        confirmed_player_control = [None] * total_frames

        target_seq = (
            raw_player_control if control_level == "player" else raw_team_control
        )

        i = 0
        while i < total_frames:
            val = target_seq[i]
            if val is None or raw_team_control[i] is None:
                i += 1
                continue

            start_idx = i
            while (
                i < total_frames
                and target_seq[i] == val
                and raw_team_control[i] is not None
            ):
                i += 1
            streak_len = i - start_idx

            if streak_len >= min_consecutive_frames:
                for k in range(start_idx, i):
                    confirmed_team_control[k] = raw_team_control[k]
                    confirmed_player_control[k] = raw_player_control[k]

        #compute the number of frames each team has control
        frames_team_1 = sum(1 for t in confirmed_team_control if t == 1)
        frames_team_2 = sum(1 for t in confirmed_team_control if t == 2)

        # formula: time = frames / frame_rate
        time_team_1 = frames_team_1 / frame_rate
        time_team_2 = frames_team_2 / frame_rate

        # compute possession ratio and percentage
        total_time = time_team_1 + time_team_2
        if total_time > 0:
            ratio_team_1 = time_team_1 / total_time
            ratio_team_2 = time_team_2 / total_time
        else:
            ratio_team_1 = 0.0
            ratio_team_2 = 0.0

        percent_team_1 = ratio_team_1 * 100.0
        percent_team_2 = ratio_team_2 * 100.0

        # compute cumulative possession over time
        cum_frames_1 = 0
        cum_frames_2 = 0
        cumulative_possession = []
        for f in range(total_frames):
            if confirmed_team_control[f] == 1:
                cum_frames_1 += 1
            elif confirmed_team_control[f] == 2:
                cum_frames_2 += 1

            c_t1 = cum_frames_1 / frame_rate
            c_t2 = cum_frames_2 / frame_rate
            c_total = c_t1 + c_t2
            c_r1 = (c_t1 / c_total * 100.0) if c_total > 0 else 50.0
            c_r2 = (c_t2 / c_total * 100.0) if c_total > 0 else 50.0

            cumulative_possession.append({
                "time_team_1": c_t1,
                "time_team_2": c_t2,
                "percent_team_1": c_r1,
                "percent_team_2": c_r2,
                "current_team": confirmed_team_control[f],
                "current_player": confirmed_player_control[f],
            })

        return {
            "team_A": {
                "team_id": 1,
                "team_name": "Team A",
                "frames": frames_team_1,
                "time_seconds": round(time_team_1, 2),
                "possession_ratio": round(ratio_team_1, 4),
                "possession_percentage": round(percent_team_1, 2),
            },
            "team_B": {
                "team_id": 2,
                "team_name": "Team B",
                "frames": frames_team_2,
                "time_seconds": round(time_team_2, 2),
                "possession_ratio": round(ratio_team_2, 4),
                "possession_percentage": round(percent_team_2, 2),
            },
            1: {
                "team_id": 1,
                "team_name": "Team A",
                "frames": frames_team_1,
                "time_seconds": round(time_team_1, 2),
                "possession_ratio": round(ratio_team_1, 4),
                "possession_percentage": round(percent_team_1, 2),
            },
            2: {
                "team_id": 2,
                "team_name": "Team B",
                "frames": frames_team_2,
                "time_seconds": round(time_team_2, 2),
                "possession_ratio": round(ratio_team_2, 4),
                "possession_percentage": round(percent_team_2, 2),
            },
            "frame_rate": frame_rate,
            "max_distance": max_distance,
            "min_consecutive_frames": min_consecutive_frames,
            "team_control_per_frame": confirmed_team_control,
            "player_control_per_frame": confirmed_player_control,
            "cumulative_possession": cumulative_possession,
        }

    def draw_ball_possession(
        self,
        frame: np.ndarray,
        frame_num: int,
        possession_info: dict,
        team_colors: dict = None,
    ) -> np.ndarray:
        if frame is None or possession_info is None:
            return frame

        cum_list = possession_info.get("cumulative_possession", [])
        if frame_num < len(cum_list):
            curr_data = cum_list[frame_num]
            pct_a = curr_data["percent_team_1"]
            pct_b = curr_data["percent_team_2"]
            t_a = curr_data["time_team_1"]
            t_b = curr_data["time_team_2"]
            active_team = curr_data["current_team"]
        else:
            team_a_info = possession_info.get("team_A", {})
            team_b_info = possession_info.get("team_B", {})
            pct_a = team_a_info.get("possession_percentage", 50.0)
            pct_b = team_b_info.get("possession_percentage", 50.0)
            t_a = team_a_info.get("time_seconds", 0.0)
            t_b = team_b_info.get("time_seconds", 0.0)
            active_team = None

        color_a = (255, 255, 255)
        color_b = (0, 0, 255)
        if team_colors:
            if 1 in team_colors:
                color_a = tuple(map(int, team_colors[1]))
            if 2 in team_colors:
                color_b = tuple(map(int, team_colors[2]))

        h, w = frame.shape[:2]
        box_w, box_h = 350, 80
        x1 = w - box_w - 20
        y1 = 15
        x2 = x1 + box_w
        y2 = y1 + box_h

        # draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "BALL POSSESSION",
            (x1 + 90, y1 + 20),
            font,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        col_text_a = (0, 255, 0) if active_team == 1 else (255, 255, 255)
        col_text_b = (0, 255, 0) if active_team == 2 else (255, 255, 255)

        text_a = f"Team A: {pct_a:.1f}% ({t_a:.1f}s)"
        text_b = f"Team B: {pct_b:.1f}% ({t_b:.1f}s)"
        cv2.putText(
            frame, text_a, (x1 + 10, y1 + 42), font, 0.42, col_text_a, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, text_b, (x1 + 180, y1 + 42), font, 0.42, col_text_b, 1, cv2.LINE_AA
        )

        # draw possession bar
        bar_x1 = x1 + 10
        bar_y1 = y1 + 52
        bar_w = box_w - 20
        bar_h = 16
        bar_x2 = bar_x1 + bar_w
        bar_y2 = bar_y1 + bar_h

        split_w = (
            int(bar_w * (pct_a / 100.0)) if (pct_a + pct_b) > 0 else int(bar_w / 2)
        )
        split_w = max(0, min(bar_w, split_w))

        if split_w > 0:
            cv2.rectangle(
                frame, (bar_x1, bar_y1), (bar_x1 + split_w, bar_y2), color_a, -1
            )
        if bar_w - split_w > 0:
            cv2.rectangle(
                frame, (bar_x1 + split_w, bar_y1), (bar_x2, bar_y2), color_b, -1
            )
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1)

        return frame

    def draw_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        tracks: dict,
        camera_movement: list = None,
        camera_movement_tracker: CameraMovement = None,
        possession_info: dict = None,
        team_colors: dict = None,
    ) -> np.ndarray:
        player_key = "players" if "players" in tracks else "player"
        player_tracks_list = tracks.get(player_key, [])

        player_tracks = (
            player_tracks_list[frame_num]
            if frame_num < len(player_tracks_list)
            else {}
        )
        ball_tracks = (
            tracks.get("ball", [])[frame_num]
            if "ball" in tracks and frame_num < len(tracks["ball"])
            else {}
        )
        referee_tracks = (
            tracks.get("referee", [])[frame_num]
            if "referee" in tracks and frame_num < len(tracks["referee"])
            else {}
        )
        keypoints_tracks = (
            tracks.get("keypoints", [])[frame_num]
            if "keypoints" in tracks and frame_num < len(tracks["keypoints"])
            else {}
        )

        # draw player elipse
        for track_id, track in player_tracks.items():
            bbox = track["bbox"]
            assigned_color = track.get("team_color", (0, 0, 255))
            self.draw_elipse(frame, bbox, assigned_color, track_id)

        # draw ball indicator
        for track_id, track in ball_tracks.items():
            bbox = track["bbox"]
            self.draw_ball_indicator(frame, bbox, (0, 255, 255))

        # draw keypoints
        for track_id, track in keypoints_tracks.items():
            # Use camera compensated coordinates when available; otherwise
            # draw the original detected point.
            position = track.get("position")
            if position is None:
                position = track.get("position_adjusted", track.get("position"))
            if position is not None:
                self.draw_keypoint(frame, position, color=(0, 0, 255), radius=6)

        # draw referee elipse
        for track_id, track in referee_tracks.items():
            bbox = track["bbox"]
            self.draw_elipse(frame, bbox, (255, 0, 0))

        # draw camera movement overlay if available
        if camera_movement is not None and camera_movement_tracker is not None:
            frame = camera_movement_tracker.draw_camera_movement(frame, camera_movement)

        # draw ball possession overlay if available
        if possession_info is not None:
            frame = self.draw_ball_possession(
                frame, frame_num, possession_info, team_colors=team_colors
            )

        return frame

    def draw_annotations(
        self,
        video_frames,
        tracks: dict,
        camera_movements: list = None,
        possession_info: dict = None,
        team_colors: dict = None,
    ):
        camera_movement_tracker = CameraMovement() if camera_movements else None

        if isinstance(video_frames, list):
            for frame_num, frame in enumerate(video_frames):
                c_mov = (
                    camera_movements[frame_num]
                    if camera_movements and frame_num < len(camera_movements)
                    else None
                )
                self.draw_frame(
                    frame,
                    frame_num,
                    tracks,
                    c_mov,
                    camera_movement_tracker,
                    possession_info=possession_info,
                    team_colors=team_colors,
                )
            return video_frames
        elif hasattr(video_frames, "__next__") or (
            hasattr(video_frames, "__iter__")
            and not isinstance(video_frames, np.ndarray)
        ):
            def generator():
                for frame_num, frame in enumerate(video_frames):
                    c_mov = (
                        camera_movements[frame_num]
                        if camera_movements and frame_num < len(camera_movements)
                        else None
                    )
                    self.draw_frame(
                        frame,
                        frame_num,
                        tracks,
                        c_mov,
                        camera_movement_tracker,
                        possession_info=possession_info,
                        team_colors=team_colors,
                    )
                    yield frame

            return generator()
        else:
            c_mov = camera_movements[0] if camera_movements else None
            return self.draw_frame(
                video_frames,
                0,
                tracks,
                c_mov,
                camera_movement_tracker,
                possession_info=possession_info,
                team_colors=team_colors,
            )

    def annotate_and_save_video(
        self,
        input_video_path: str="Personal_Project/Football_Analyst_Project/Inputs/Video_Project.mp4",
        output_video_path: str="Personal_Project/Football_Analyst_Project/Outputs/Match_Video_R.mp4",
        tracks: dict = None,
        measurement=None,
        possession_info: dict = None,
        draw_possession: bool = True,
    ) -> None:
        input_video_path = input_video_path
        camera_movement_tracker = CameraMovement()
        camera_movements = camera_movement_tracker.get_camera_movement(
            input_video_path,
            use_stub=True,
            stub_path="Personal_Project/Football_Analyst_Project/Stubs/camera_movement_stub.pkl",
        )
        camera_movement_tracker.add_camera_position_to_tracks(tracks, camera_movements)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Khong the mo: {input_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        # Tự động tính toán kiểm soát bóng nếu được yêu cầu vẽ và chưa được cung cấp
        if draw_possession and possession_info is None:
            possession_info = self.calculate_ball_possession(tracks, frame_rate=fps)

        # Trích xuất màu áo của 2 đội từ tracks nếu có
        team_colors = {}
        player_key = "players" if "players" in tracks else "player"
        for p_dict in tracks.get(player_key, []):
            for p_info in p_dict.values():
                t = p_info.get("team")
                c = p_info.get("team_color")
                if t is not None and c is not None and t not in team_colors:
                    team_colors[t] = c
            if len(team_colors) >= 2:
                break

        def frame_stream():
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                c_mov = (
                    camera_movements[frame_num]
                    if frame_num < len(camera_movements)
                    else None
                )
                frame = self.draw_frame(
                    frame,
                    frame_num,
                    tracks,
                    c_mov,
                    camera_movement_tracker,
                    possession_info=possession_info,
                    team_colors=team_colors,
                )

                # draw distance and velocity if measurement is provided
                if measurement is not None:
                    frame = measurement.draw_distance_and_velocity(
                        frame, tracks, frame_num
                    )

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
        tracks["ball"] = [
            {1: {"bbox": row.values.tolist()}}
            for _, row in ball_df_interpolated.iterrows()
        ]
        return tracks

    def draw_keypoint(
        self, frame: any, position: list, color: any = (0, 0, 255), radius: int = 6
    ) -> any:
        x, y = int(position[0]), int(position[1])
        # Viền ngoài trắng để tương phản mạnh trên nền cỏ sân bóng / màu áo
        cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), -1)
        # Tâm tròn màu đỏ rực rỡ nổi bật
        cv2.circle(frame, (x, y), radius, color, -1)
        return frame

    def draw_elipse(
        self, frame: any, bbox: list, color: any, track_id: int = None
    ) -> any:
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
                frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness=2,
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
