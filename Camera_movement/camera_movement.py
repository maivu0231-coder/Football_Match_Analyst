import os
import pickle
import cv2
import numpy as np


class CameraMovement:
    def __init__(self, frame=None):
        self.minimum_movement_frame = 5
        self.optical_flow_params = dict(
            winSize=(15, 15),#win size for searching the neighborhood of a point
            maxLevel=2,# the number of pyramid layers including the initial image
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)#the number of iterations and the desired accuracy for the iterative search algorithm
        )
        self.features_params = None
        if frame is not None:
            self._init_features_params(frame)

    def _init_features_params(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask_features = np.zeros_like(frame_gray)
        mask_features[:, 0:100]= 1       # 10% biên trái
        mask_features[:, -100:] = 1        # 10% biên phải

        self.features_params = dict(
            mask=mask_features,
            maxCorners=100,# maximum number of corners to return
            qualityLevel=0.3,# minimum quality level of corners to be considered
            minDistance=3,# minimum distance between corners to be considered
            blockSize=7# size of the block to compute the corner quality
        )

    def get_camera_movement(
        self, 
        video_input, 
        use_stub: bool = False, 
        stub_path: str = "Personal_Project/Football_Analyst_Project/Stubs/camera_movement_stub.pkl"
    ):

        if use_stub and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)
        #check if video_input is a path or a list of frames
        is_path = isinstance(video_input, str)
        if is_path:
            cap = cv2.VideoCapture(video_input)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                return []
            self._init_features_params(first_frame)
            old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            total_frames = len(video_input)
            if total_frames == 0:
                return []
            self._init_features_params(video_input[0])
            old_gray = cv2.cvtColor(video_input[0], cv2.COLOR_BGR2GRAY)
        #inilize camera_movements list with first frame movement as (0,0)
        camera_movements = [[0.0, 0.0]] 
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features_params)

        print(f"Bắt đầu tính toán Camera Movement ({total_frames} frames)...")
        frame_idx = 1
        while True:
            if is_path:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                if frame_idx >= total_frames:
                    break
                frame = video_input[frame_idx]

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            camera_movement_x, camera_movement_y = 0.0, 0.0

            if old_features is not None and len(old_features) > 0:
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, old_features, None, **self.optical_flow_params
                )

                if new_features is not None and status is not None:
                    good_new = new_features[status == 1]
                    good_old = old_features[status == 1]

                    if len(good_new) > 0:
                        diffs = good_new - good_old
                        distances = np.linalg.norm(diffs, axis=1)
                        # Chọn chỉ số của vector chuyển động có khoảng cách lớn nhất
                        max_idx = int(np.argmax(distances))

                        if distances[max_idx] > self.minimum_movement_frame:
                            camera_movement_x = float(diffs[max_idx][0])
                            camera_movement_y = float(diffs[max_idx][1])
                            old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features_params)
                        else:
                            old_features = good_new.reshape(-1, 1, 2)

            camera_movements.append([camera_movement_x, camera_movement_y])
            old_gray = frame_gray.copy()
            frame_idx += 1

            if total_frames > 0 and frame_idx % 100 == 0:
                print(f"\rTiến độ: {(frame_idx / total_frames) * 100:.1f}% ({frame_idx}/{total_frames})", end="", flush=True)

        if is_path:
            cap.release()
        print("\nHoàn thành tính toán Camera Movement!")

        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, "wb") as f:
            pickle.dump(camera_movements, f)

        return camera_movements

    def draw_camera_movement(self, frame: np.ndarray, camera_movement_per_frame: list) -> np.ndarray:

        overlay_w, overlay_h = 320, 75
        x1, y1, x2, y2 = 10, 10, 10 + overlay_w, 10 + overlay_h
        sub_img = frame[y1:y2, x1:x2]
        white_rect = np.full(sub_img.shape, 255, dtype=np.uint8)
        blended = cv2.addWeighted(sub_img, 0.4, white_rect, 0.6, 0)
        frame[y1:y2, x1:x2] = blended
        #show camera movement values on the overlay
        x_m = camera_movement_per_frame[0]
        y_m = camera_movement_per_frame[1]

        cv2.putText(frame, f"Camera Movement:", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f"X: {x_m:+.2f} px  |  Y: {y_m:+.2f} px", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)

        return frame

    def add_camera_position_to_tracks(self, tracks: dict, camera_movements: list):
        # 1. Tính tổng cộng dồn chuyển động camera từ frame 0
        accumulated_x, accumulated_y = 0.0, 0.0
        cumulative_movements = []
        for mov in camera_movements:
            accumulated_x += mov[0]
            accumulated_y += mov[1]
            cumulative_movements.append([accumulated_x, accumulated_y])

        # 2. Bù trừ theo độ dời cộng dồn
        for obj, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if frame_num >= len(cumulative_movements):
                    break
                camera_movement = cumulative_movements[frame_num]
                for track_id, track_data in track.items():
                    position = track_data.get("position")
                    if position is not None:
                        position_adjusted = [
                            position[0] - camera_movement[0],
                            position[1] - camera_movement[1],
                        ]
                        tracks[obj][frame_num][track_id]["position_adjusted"] = (
                            position_adjusted
                        )