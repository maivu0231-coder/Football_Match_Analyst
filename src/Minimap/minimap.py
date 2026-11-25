from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Optional, Tuple

import cv2
import numpy as np


class MiniMap:

    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0
    PENALTY_AREA_DEPTH = 16.5
    PENALTY_AREA_WIDTH = 40.32
    GOAL_AREA_DEPTH = 5.5
    GOAL_AREA_WIDTH = 18.32
    PENALTY_SPOT_DISTANCE = 11.0
    CENTER_CIRCLE_RADIUS = 9.15
    NUM_LANDMARKS = 32

    # Standard pitch coordinates, indexed by the proposed 0..31 layout.
    DEFAULT_LANDMARKS_METERS = {
        0: (0.0, 0.0), 1: (0.0, 13.84), 2: (0.0, 24.84),
        3: (0.0, 43.16), 4: (0.0, 54.16), 5: (0.0, 68.0),
        6: (16.5, 13.84), 7: (16.5, 24.84), 8: (16.5, 43.16),
        9: (16.5, 54.16), 10: (5.5, 24.84), 11: (5.5, 43.16),
        12: (11.0, 34.0),
        13: (52.5, 0.0), 14: (52.5, 24.85), 15: (43.35, 34.0),
        16: (61.65, 34.0), 17: (52.5, 43.15), 18: (52.5, 68.0),
        19: (105.0, 0.0), 20: (105.0, 13.84), 21: (105.0, 24.84),
        22: (105.0, 43.16), 23: (105.0, 54.16), 24: (105.0, 68.0),
        25: (88.5, 13.84), 26: (88.5, 24.84), 27: (88.5, 43.16),
        28: (88.5, 54.16), 29: (99.5, 24.84), 30: (99.5, 43.16),
        31: (94.0, 34.0),
    }

    def __init__(
        self,
        width: int = 420,
        height: int = 280,
        margin: int = 14,
        target_points: Optional[Mapping[int, Tuple[float, float]] | np.ndarray] = None,
        keypoint_to_target: Optional[Mapping[int, int]] = None,
        background_color: Tuple[int, int, int] = (45, 125, 45),
        line_color: Tuple[int, int, int] = (245, 245, 245),
        enable_ball_trail: bool = True,
        ball_trail_length: int = 8,
        homography_smoothing_alpha: float = 0.5,
    ):
        if width <= 2 * margin or height <= 2 * margin:
            raise ValueError("Mini-map dimensions must be larger than its margins")
        self.width = int(width)
        self.height = int(height)
        self.margin = int(margin)
        self.background_color = tuple(map(int, background_color))
        self.line_color = tuple(map(int, line_color))
        self.enable_ball_trail = bool(enable_ball_trail)
        self.ball_trail_length = int(ball_trail_length)
        self._ball_history = deque(maxlen=self.ball_trail_length)
        # 0.5: balance between temporal stability and camera-motion response.
        self.homography_smoothing_alpha = float(
            np.clip(homography_smoothing_alpha, 0.0, 1.0)
        )

        scale = min(
            (self.width - 2 * self.margin) / self.PITCH_LENGTH,
            (self.height - 2 * self.margin) / self.PITCH_WIDTH,
        )
        self.scale = scale
        self.pitch_pixel_length = self.PITCH_LENGTH * scale
        self.pitch_pixel_width = self.PITCH_WIDTH * scale
        self.pitch_left = (self.width - self.pitch_pixel_length) / 2.0
        self.pitch_top = (self.height - self.pitch_pixel_width) / 2.0

        if target_points is None:
            self.target_points = {
                point_id: self._meters_to_pixel(*xy)
                for point_id, xy in self.DEFAULT_LANDMARKS_METERS.items()
            }
        elif isinstance(target_points, Mapping):
            self.target_points = {
                int(point_id): tuple(map(float, xy))
                for point_id, xy in target_points.items()
            }
        else:
            points = np.asarray(target_points, dtype=np.float64)
            if points.shape != (self.NUM_LANDMARKS, 2):
                raise ValueError("target_points array must have shape (32, 2)")
            self.target_points = {
                point_id: tuple(map(float, points[point_id]))
                for point_id in range(self.NUM_LANDMARKS)
            }

        self.keypoint_to_target = {
            13: 13, 14: 14, 15: 17, 16: 18, 17: 25, 18: 26,
            19: 27, 20: 28, 21: 30, 22: 29, 23: 31, 24: 19,
            25: 20, 26: 21, 27: 22, 28: 23, 30: 15, 31: 16,
        }
        if keypoint_to_target:
            self.keypoint_to_target.update(
                {int(k): int(v) for k, v in keypoint_to_target.items()}
            )
        self.last_homography_info = None
        self._last_valid_homography = None
        self._last_valid_homography_frame = None
        self.homography_hold_frames = 5
        self._smoothed_positions = {}
        self._ball_smoothed_velocity = {}

    @staticmethod
    def _blend_homography(current, previous, current_weight: float = 0.35):
        """Temporally blend successive valid homographies without scale jumps."""
        current = np.asarray(current, dtype=np.float64)
        if current.shape != (3, 3) or not np.isfinite(current).all():
            return None

        def normalize(matrix):
            scale = matrix[2, 2]
            if abs(scale) < 1e-9:
                scale = np.linalg.norm(matrix)
            if abs(scale) < 1e-12:
                return None
            return matrix / scale

        current = normalize(current)
        if current is None:
            return None
        if previous is None:
            return current

        previous = normalize(np.asarray(previous, dtype=np.float64))
        if previous is None or previous.shape != (3, 3) or not np.isfinite(previous).all():
            return current
        # H and -H represent the same projective transform; align signs before
        # averaging so equivalent matrices cannot cancel each other out.
        if float(np.sum(current * previous)) < 0.0:
            current = -current
        weight = float(np.clip(current_weight, 0.0, 1.0))
        blended = (1.0 - weight) * previous + weight * current
        blended = normalize(blended)
        if blended is None or abs(float(np.linalg.det(blended))) < 1e-10:
            return current
        return blended

    def _meters_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        px = self.pitch_left + x / self.PITCH_LENGTH * self.pitch_pixel_length
        py = self.pitch_top + y / self.PITCH_WIDTH * self.pitch_pixel_width
        return float(px), float(py)

    def _pixel_to_meters(self, point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = map(float, point)
        pitch_x = (x - self.pitch_left) / self.pitch_pixel_length * self.PITCH_LENGTH
        pitch_y = (y - self.pitch_top) / self.pitch_pixel_width * self.PITCH_WIDTH
        return float(pitch_x), float(pitch_y)

    def _pitch_point(self, x: float, y: float) -> Tuple[int, int]:
        px, py = self._meters_to_pixel(x, y)
        return int(round(px)), int(round(py))

    def _draw_pitch_line(self, image, p1, p2, thickness=2):
        cv2.line(
            image,
            self._pitch_point(*p1),
            self._pitch_point(*p2),
            self.line_color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_minimap(self) -> np.ndarray:
        image = np.full(
            (self.height, self.width, 3), self.background_color, dtype=np.uint8
        )
        length, width = self.PITCH_LENGTH, self.PITCH_WIDTH
        half_width = width / 2.0
        pa_top = (width - self.PENALTY_AREA_WIDTH) / 2.0
        pa_bottom = width - pa_top
        ga_top = (width - self.GOAL_AREA_WIDTH) / 2.0
        ga_bottom = width - ga_top

        for start, end in (
            ((0.0, 0.0), (length, 0.0)),
            ((length, 0.0), (length, width)),
            ((length, width), (0.0, width)),
            ((0.0, width), (0.0, 0.0)),
            ((length / 2.0, 0.0), (length / 2.0, width)),
        ):
            self._draw_pitch_line(image, start, end)

        for goal_x, direction in ((0.0, 1.0), (length, -1.0)):
            pa_x = goal_x + direction * self.PENALTY_AREA_DEPTH
            ga_x = goal_x + direction * self.GOAL_AREA_DEPTH
            for start, end in (
                ((goal_x, pa_top), (pa_x, pa_top)),
                ((pa_x, pa_top), (pa_x, pa_bottom)),
                ((pa_x, pa_bottom), (goal_x, pa_bottom)),
                ((goal_x, ga_top), (ga_x, ga_top)),
                ((ga_x, ga_top), (ga_x, ga_bottom)),
                ((ga_x, ga_bottom), (goal_x, ga_bottom)),
            ):
                self._draw_pitch_line(image, start, end)
            spot_x = goal_x + direction * self.PENALTY_SPOT_DISTANCE
            cv2.circle(image, self._pitch_point(spot_x, half_width), 2, self.line_color, -1, cv2.LINE_AA)

        center = self._pitch_point(length / 2.0, half_width)
        circle_radius = int(round(
            self.CENTER_CIRCLE_RADIUS / width * self.pitch_pixel_width
        ))
        cv2.circle(image, center, circle_radius, self.line_color, 2, cv2.LINE_AA)
        cv2.circle(image, center, 2, self.line_color, -1, cv2.LINE_AA)
        return image

    @staticmethod
    def _position_and_confidence(value):
        if isinstance(value, Mapping):
            position = value.get("position")
            confidence = float(value.get("confidence", 1.0))
        else:
            position, confidence = value, 1.0
        if position is None:
            return None, confidence
        point = np.asarray(position, dtype=np.float64).reshape(-1)
        if point.size < 2 or not np.isfinite(point[:2]).all():
            return None, confidence
        return point[:2], confidence

    def calculate_homography(
        self,
        image_keypoints: Mapping[int, object],
        confidence_threshold: float = 0.1,
        ransac_threshold: float = 14.0,
        min_inliers: int = 4,
        min_inlier_ratio: float = 0.15,
        keypoint_to_target: Optional[Mapping[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], dict]:
        id_mapping = dict(self.keypoint_to_target)
        if keypoint_to_target is not None:
            id_mapping.update({int(k): int(v) for k, v in keypoint_to_target.items()})

        source, destination, model_ids, target_ids = [], [], [], []
        for raw_id, value in image_keypoints.items():
            try:
                model_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if id_mapping and model_id not in id_mapping:
                continue
            target_id = int(id_mapping[model_id] if id_mapping else model_id)
            if target_id not in self.target_points:
                continue
            position, confidence = self._position_and_confidence(value)
            if position is None or confidence < confidence_threshold:
                continue
            source.append(position)
            destination.append(self.target_points[target_id])
            model_ids.append(model_id)
            target_ids.append(target_id)

        info = {
            "matched_model_ids": model_ids,
            "matched_target_ids": target_ids,
            "inlier_model_ids": [],
            "inlier_target_ids": [],
            "outlier_model_ids": [],
            "outlier_target_ids": [],
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "success": False,
            "usable": False,
        }
        if len(source) < 4:
            info["reason"] = "fewer than four valid correspondences"
            self.last_homography_info = info
            return None, info

        src = np.asarray(source, dtype=np.float64)
        dst = np.asarray(destination, dtype=np.float64)
        if np.linalg.matrix_rank(src - src.mean(axis=0)) < 2:
            info["reason"] = "source points are collinear or clustered"
            self.last_homography_info = info
            return None, info
        if np.linalg.matrix_rank(dst - dst.mean(axis=0)) < 2:
            info["reason"] = "target points are collinear or clustered"
            self.last_homography_info = info
            return None, info

        homography, mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_threshold),
        )
        if homography is None or mask is None or not np.isfinite(homography).all():
            info["reason"] = "RANSAC could not estimate a homography"
            self.last_homography_info = info
            return None, info

        inliers = mask.reshape(-1).astype(bool)
        info["inlier_model_ids"] = [i for i, keep in zip(model_ids, inliers) if keep]
        info["inlier_target_ids"] = [i for i, keep in zip(target_ids, inliers) if keep]
        info["outlier_model_ids"] = [i for i, keep in zip(model_ids, inliers) if not keep]
        info["outlier_target_ids"] = [i for i, keep in zip(target_ids, inliers) if not keep]
        info["inlier_count"] = int(inliers.sum())
        info["inlier_ratio"] = (
            info["inlier_count"] / len(inliers) if len(inliers) else 0.0
        )

        inlier_target_points = dst[inliers].astype(np.float32)
        hull_area = (
            float(cv2.contourArea(cv2.convexHull(inlier_target_points)))
            if len(inlier_target_points) >= 3
            else 0.0
        )
        pitch_area = self.pitch_pixel_length * self.pitch_pixel_width
        info["inlier_target_coverage"] = hull_area / pitch_area if pitch_area else 0.0
        inlier_source_points = src[inliers].astype(np.float32)
        source_inliers_are_2d = (
            len(inlier_source_points) >= 4
            and np.linalg.matrix_rank(
                inlier_source_points - inlier_source_points.mean(axis=0)
            ) >= 2
        )
        target_inliers_are_2d = (
            len(inlier_target_points) >= 4
            and np.linalg.matrix_rank(
                inlier_target_points - inlier_target_points.mean(axis=0)
            ) >= 2
        )
        info["usable"] = source_inliers_are_2d and target_inliers_are_2d
        info["success"] = (
            info["usable"]
            and info["inlier_count"] >= int(min_inliers)
            and info["inlier_ratio"] >= float(min_inlier_ratio)
            and info["inlier_target_coverage"] >= 0.01
        )
        if not info["usable"]:
            info["reason"] = "RANSAC inliers are insufficient or collinear"
            homography = None
        elif not info["success"]:
            info["reason"] = "weak RANSAC fit; displaying with extra smoothing"
        self.last_homography_info = info
        return homography, info

    def transform_point(
        self, point, homography: np.ndarray, is_ball: bool = False
    ) -> Optional[Tuple[float, float]]:
        if homography is None or point is None:
            return None
        point_array = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        projected = cv2.perspectiveTransform(point_array, homography.astype(np.float64))
        x, y = map(float, projected.reshape(2))
        if not np.isfinite([x, y]).all():
            return None

        # Cho phép biên rộng hơn cho bóng khi bay cao, tránh bóng biến mất
        margin = 25.0 if is_ball else 8.0
        if (
            x < self.pitch_left - margin
            or x > self.pitch_left + self.pitch_pixel_length + margin
            or y < self.pitch_top - margin
            or y > self.pitch_top + self.pitch_pixel_width + margin
        ):
            return None
        return x, y

    @staticmethod
    def _track_image_position(track, object_type: str):
        """Lấy vị trí 2D của track trên ảnh:
        - Với bóng: ưu tiên tâm bóng (position hoặc tâm bbox) để triệt tiêu rung giật viền cỏ.
        - Với cầu thủ/trọng tài: lấy đáy bbox (vị trí bàn chân chạm mặt sân).
        """
        position = track.get("position")
        if position is not None and len(position) >= 2:
            return float(position[0]), float(position[1])

        bbox = track.get("bbox")
        if bbox is None or len(bbox) < 4:
            return None

        x1, y1, x2, y2 = map(float, bbox[:4])
        if object_type == "ball":
            # Dùng tâm bbox cho quả bóng
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        # Dùng đáy bbox cho bàn chân cầu thủ
        return ((x1 + x2) / 2.0, y2)

    def project_tracks(
        self,
        tracks: dict,
        frame_index: int,
        homography: np.ndarray,
        smoothing: bool = True,
    ) -> dict:
        """Chiếu toạ độ cầu thủ, trọng tài và bóng lên minimap pixel."""
        projected = {"player": {}, "referee": {}, "ball": {}}
        if homography is None:
            return projected

        player_key = "players" if "players" in tracks else "player"
        for object_type, key in (
            ("player", player_key),
            ("referee", "referee"),
            ("ball", "ball"),
        ):
            frames = tracks.get(key, [])
            if frame_index < 0 or frame_index >= len(frames):
                continue
            for track_id, track in frames[frame_index].items():
                stored_map_position = track.get("position_minimap")
                if stored_map_position is not None:
                    map_position = tuple(map(float, stored_map_position))
                    used_stored_position = True
                else:
                    image_position = self._track_image_position(track, object_type)
                    map_position = self.transform_point(
                        image_position, homography, is_ball=(object_type == "ball")
                    )
                    used_stored_position = False

                if map_position is not None:
                    # Giữ phép chiếu bóng theo frame gốc. Làm mượt nhân quả ở
                    # đây có thể biến một lỗi homography đơn frame thành vệt
                    # bóng kéo dài; bóng được lọc hậu kỳ trên toàn chuỗi.
                    if smoothing and not used_stored_position and object_type != "ball":
                        map_position = self._smooth_track_position(
                            object_type, track_id, map_position, frame_index
                        )
                    projected[object_type][track_id] = {
                        "position": map_position,
                        "team_color": track.get("team_color"),
                        "team": track.get("team"),
                        "confidence": track.get("confidence", 1.0),
                    }

        return projected

    def _smooth_track_position(self, object_type, track_id, position, frame_index):
        """Làm mượt toạ độ thích ứng theo loại đối tượng."""
        key = (object_type, track_id)
        current = np.asarray(position, dtype=np.float32)
        previous_state = self._smoothed_positions.get(key)
        if previous_state is None:
            self._smoothed_positions[key] = (frame_index, current)
            if object_type == "ball":
                self._ball_smoothed_velocity[key] = np.zeros(2, dtype=np.float32)
            return tuple(map(float, current))

        previous_frame, previous = previous_state
        elapsed = max(1, int(frame_index) - int(previous_frame))
        if elapsed > 10:
            self._smoothed_positions[key] = (frame_index, current)
            if object_type == "ball":
                self._ball_smoothed_velocity[key] = np.zeros(2, dtype=np.float32)
            return tuple(map(float, current))

        if object_type == "ball":
            # Giới hạn vận tốc vật lý tối đa của bóng: ~40 m/s (~144 km/h)
            max_ball_speed_mps = 40.0
            max_step_pixel = max_ball_speed_mps * (self.scale / 30.0) * elapsed * 1.5

            velocity = self._ball_smoothed_velocity.get(key, np.zeros(2, dtype=np.float32))
            predicted = previous + velocity * elapsed
            residual = current - predicted
            residual_size = float(np.linalg.norm(residual))

            if residual_size <= 2.0:
                # Vùng chết (deadband): triệt tiêu hoàn toàn rung lắc nhỏ khi bóng lăn chậm/đứng yên
                smoothed = predicted
                velocity *= 0.85
            elif residual_size <= 12.0:
                # Bóng lăn bình thường: làm mượt nhẹ
                smoothed = predicted + 0.40 * residual
                velocity = 0.5 * velocity + 0.5 * ((smoothed - previous) / elapsed)
            else:
                # Bóng bị sút hoặc chuyền nhanh: phản hồi nhạy bén, chống trễ lag
                correction = residual.copy()
                if residual_size > max_step_pixel:
                    correction *= (max_step_pixel / residual_size)
                smoothed = predicted + 0.80 * correction
                velocity = 0.3 * velocity + 0.7 * ((smoothed - previous) / elapsed)

            actual_step = smoothed - previous
            step_size = float(np.linalg.norm(actual_step))
            if step_size > max_step_pixel:
                smoothed = previous + actual_step * (max_step_pixel / step_size)

            self._ball_smoothed_velocity[key] = velocity.astype(np.float32)
        else:
            # Cầu thủ di chuyển chậm, quán tính lớn
            alpha = 0.20
            delta = current - previous
            distance = float(np.linalg.norm(delta))
            max_step = 6.0 * elapsed
            if distance > max_step:
                current = previous + delta * (max_step / distance)
            smoothed = previous * (1.0 - alpha) + current * alpha

        self._smoothed_positions[key] = (frame_index, smoothed)
        return tuple(map(float, smoothed))

    def _interpolate_missing_ball_detections(self, tracks: dict, max_gap: int = 8):
        """Lấp đầy các frame bị mất dấu bóng (missed detections) qua nội suy."""
        ball_frames = tracks.get("ball", [])
        if not ball_frames:
            return

        # Gom nhóm toạ độ bóng đã biết
        known_ball_frames = []
        for f_idx, f_data in enumerate(ball_frames):
            for b_id, b_info in f_data.items():
                if "position_minimap" in b_info:
                    known_ball_frames.append(
                        (f_idx, b_id, np.asarray(b_info["position_minimap"], dtype=np.float64))
                    )
                    break

        if len(known_ball_frames) < 2:
            return

        # Tìm và nội suy các khoảng trống (gap <= max_gap)
        for i in range(len(known_ball_frames) - 1):
            f_start, b_id, p_start = known_ball_frames[i]
            f_end, end_ball_id, p_end = known_ball_frames[i + 1]
            gap = f_end - f_start

            # Never invent a cross-pitch flight across a missing-detection gap.
            # At this map scale, 12 px/frame is already a generous ball speed.
            speed_per_frame = float(np.linalg.norm(p_end - p_start)) / max(1, gap)
            if (
                1 < gap <= max_gap
                and b_id == end_ball_id
                and speed_per_frame <= 12.0
            ):
                for step, mid_frame in enumerate(range(f_start + 1, f_end), start=1):
                    alpha = step / gap
                    interp_pos = p_start * (1.0 - alpha) + p_end * alpha
                    if mid_frame >= len(ball_frames):
                        continue
                    if b_id not in ball_frames[mid_frame]:
                        ball_frames[mid_frame][b_id] = {
                            "position_minimap": interp_pos.tolist(),
                            "position_pitch": list(self._pixel_to_meters(interp_pos)),
                            "interpolated": True,
                        }

    def add_pitch_positions_to_tracks(
        self,
        tracks: dict,
        confidence_threshold: float = 0.25,
        ransac_threshold: float = 8.0,
        min_inliers: int = 5,
        min_inlier_ratio: float = 0.4,
    ) -> dict:
        """Gán toạ độ minimap pixel và pitch meters chuẩn xác cho toàn bộ video."""
        keypoint_frames = tracks.get("keypoints", [])
        frame_count = max(
            (len(tracks.get(key, [])) for key in ("player", "players", "referee", "ball")),
            default=0,
        )
        homographies = [None] * frame_count
        last_homography = None
        last_valid_frame = None

        for frame_index in range(frame_count):
            keypoints = (
                keypoint_frames[frame_index]
                if frame_index < len(keypoint_frames)
                else {}
            )
            raw_homography, fit_info = self.calculate_homography(
                keypoints,
                confidence_threshold=confidence_threshold,
                ransac_threshold=ransac_threshold,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
            )

            if raw_homography is not None and fit_info.get("success", False):
                # Only blend a valid RANSAC fit into the temporal H state.
                homography = self._blend_homography(
                    raw_homography, last_homography
                )
                if homography is not None:
                    last_homography = homography
                    last_valid_frame = frame_index
                else:
                    homography = None
            else:
                homography = None

            # Reuse the last good H briefly when keypoints are temporarily lost.
            if homography is None and (
                last_homography is not None
                and last_valid_frame is not None
                and frame_index - last_valid_frame <= self.homography_hold_frames
            ):
                homography = last_homography.copy()
            homographies[frame_index] = homography

        first_valid = next(
            (index for index, h in enumerate(homographies) if h is not None),
            None,
        )
        if first_valid is not None:
            for frame_index in range(first_valid):
                homographies[frame_index] = homographies[first_valid]

        object_keys = {
            "player": "players" if "players" in tracks else "player",
            "referee": "referee",
            "ball": "ball",
        }
        for object_type, key in object_keys.items():
            for frame_tracks in tracks.get(key, []):
                for track in frame_tracks.values():
                    track.pop("position_minimap", None)
                    track.pop("position_pitch", None)

        self._smoothed_positions.clear()
        self._ball_smoothed_velocity.clear()

        # Chiếu toạ độ cho các frame hợp lệ
        for frame_index, homography in enumerate(homographies):
            if homography is None:
                continue
            projected = self.project_tracks(
                tracks,
                frame_index,
                homography,
                smoothing=True,
            )
            for object_type, key in object_keys.items():
                frames = tracks.get(key, [])
                if frame_index >= len(frames):
                    continue
                for track_id, values in projected[object_type].items():
                    track = frames[frame_index].get(track_id)
                    if track is None:
                        continue
                    map_position = values["position"]
                    track["position_minimap"] = [float(map_position[0]), float(map_position[1])]
                    track["position_pitch"] = list(self._pixel_to_meters(map_position))

        # Lấp đầy các frame bị mất dấu bóng (missed detections)
        self._interpolate_missing_ball_detections(tracks, max_gap=8)

        # Lọc mịn và xử lý quỹ đạo bóng bổng
        self._interpolate_airborne_ball_segments(tracks, object_keys["player"])
        self._smooth_ball_track_positions(tracks)

        self._homographies_by_frame = homographies
        self._last_valid_homography = None
        self._last_valid_homography_frame = None
        return tracks

    def _interpolate_airborne_ball_segments(self, tracks, player_key):
        """Khắc phục hiện tượng bóng bổng bị biến dạng do góc nhìn phối cảnh."""
        ball_frames = tracks.get("ball", [])
        by_ball_id = {}
        for frame_index, frame_tracks in enumerate(ball_frames):
            for ball_id, ball in frame_tracks.items():
                map_point = ball.get("position_minimap")
                if map_point is None:
                    continue
                by_ball_id.setdefault(ball_id, []).append({
                    "frame": frame_index,
                    "ball": ball,
                    "map": np.asarray(map_point, dtype=float),
                })

        for records in by_ball_id.values():
            if len(records) < 5:
                continue
            records.sort(key=lambda item: item["frame"])

            # Phát hiện bất thường vận tốc giả do bóng bay bổng
            speeds = [0.0]
            for i in range(1, len(records)):
                gap = max(1, records[i]["frame"] - records[i - 1]["frame"])
                dist = np.linalg.norm(records[i]["map"] - records[i - 1]["map"])
                speeds.append(dist / gap)

            # Thay thế các đoạn vọt tốc độ bất thường bằng nội suy thẳng mượt
            max_allowed_pixel_speed = 22.0
            i = 1
            while i < len(records) - 1:
                if speeds[i] > max_allowed_pixel_speed:
                    start = i - 1
                    end = i + 1
                    while end < len(records) - 1 and speeds[end] > max_allowed_pixel_speed:
                        end += 1

                    p0 = records[start]["map"]
                    p1 = records[end]["map"]
                    f0 = records[start]["frame"]
                    f1 = records[end]["frame"]

                    segment_speed = (
                        float(np.linalg.norm(p1 - p0)) / (f1 - f0)
                        if f1 > f0 else float("inf")
                    )
                    if f1 > f0 and segment_speed <= 12.0:
                        for k in range(start + 1, end):
                            ratio = (records[k]["frame"] - f0) / (f1 - f0)
                            p_interp = p0 * (1.0 - ratio) + p1 * ratio
                            records[k]["ball"]["position_minimap"] = p_interp.tolist()
                            records[k]["ball"]["position_pitch"] = list(self._pixel_to_meters(p_interp))
                    i = end + 1
                else:
                    i += 1

    def _smooth_ball_track_positions(self, tracks):
        """
        Smooth mapped ball coordinates with a constant-velocity Kalman model.

        This is an offline pass after homography projection and short-gap
        interpolation. An RTS backward pass reduces jitter without the lag
        of a forward-only filter. Gaps longer than 20 frames split the track.
        """
        by_ball_id = {}
        for frame_index, frame_tracks in enumerate(tracks.get("ball", [])):
            for ball_id, ball in frame_tracks.items():
                point = ball.get("position_minimap")
                if point is not None:
                    by_ball_id.setdefault(ball_id, []).append(
                        (frame_index, ball, np.asarray(point, dtype=np.float64))
                    )

        for records in by_ball_id.values():
            if len(records) < 3:
                continue
            records.sort(key=lambda item: item[0])

            segments = []
            segment = [records[0]]
            for record in records[1:]:
                if record[0] - segment[-1][0] > 20:
                    if len(segment) >= 3:
                        segments.append(segment)
                    segment = [record]
                else:
                    segment.append(record)
            if len(segment) >= 3:
                segments.append(segment)

            for segment in segments:
                count = len(segment)
                measurements = [item[2] for item in segment]
                filtered_states, filtered_covariances = [], []
                predicted_states = [None] * count
                predicted_covariances = [None] * count
                transitions = [None] * count

                state = np.array(
                    [measurements[0][0], measurements[0][1], 0.0, 0.0],
                    dtype=np.float64,
                )
                covariance = np.diag([16.0, 16.0, 100.0, 100.0])
                identity = np.eye(4, dtype=np.float64)
                observation = np.array(
                    [[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                )
                acceleration_variance = 12.0

                for i, (frame_idx, ball, measurement) in enumerate(segment):
                    if i == 0:
                        transition = identity.copy()
                        predicted_state = state.copy()
                        predicted_covariance = covariance.copy()
                    else:
                        dt = float(max(1, frame_idx - segment[i - 1][0]))
                        transition = np.array(
                            [[1.0, 0.0, dt, 0.0],
                             [0.0, 1.0, 0.0, dt],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]],
                            dtype=np.float64,
                        )
                        q = acceleration_variance
                        process_noise = q * np.array(
                            [[dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                             [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
                             [dt**3 / 2.0, 0.0, dt**2, 0.0],
                             [0.0, dt**3 / 2.0, 0.0, dt**2]],
                            dtype=np.float64,
                        )
                        predicted_state = transition @ state
                        predicted_covariance = (
                            transition @ covariance @ transition.T + process_noise
                        )

                    predicted_states[i] = predicted_state
                    predicted_covariances[i] = predicted_covariance
                    transitions[i] = transition

                    # Interpolated points are less reliable than YOLO detections.
                    variance = 12.0 if ball.get("interpolated", False) else 4.0
                    measurement_noise = np.diag([variance, variance])
                    innovation = measurement - observation @ predicted_state
                    innovation_covariance = (
                        observation @ predicted_covariance @ observation.T
                        + measurement_noise
                    )
                    gain = (
                        predicted_covariance @ observation.T
                        @ np.linalg.inv(innovation_covariance)
                    )
                    state = predicted_state + gain @ innovation
                    covariance = (
                        identity - gain @ observation
                    ) @ predicted_covariance

                    filtered_states.append(state.copy())
                    filtered_covariances.append(covariance.copy())

                # RTS backward pass: use future measurements to refine past states.
                smoothed_states = [item.copy() for item in filtered_states]
                for i in range(count - 2, -1, -1):
                    transition = transitions[i + 1]
                    predicted_covariance = predicted_covariances[i + 1]
                    smoother_gain = (
                        filtered_covariances[i] @ transition.T
                        @ np.linalg.pinv(predicted_covariance)
                    )
                    smoothed_states[i] = (
                        filtered_states[i]
                        + smoother_gain
                        @ (smoothed_states[i + 1] - predicted_states[i + 1])
                    )

                for (_, ball, _), smoothed_state in zip(segment, smoothed_states):
                    point = smoothed_state[:2]
                    ball["position_minimap"] = point.tolist()
                    ball["position_pitch"] = list(self._pixel_to_meters(point))

    def draw_entities(self, minimap: np.ndarray, projected_tracks: dict) -> np.ndarray:
        """Vẽ cầu thủ, trọng tài và bóng (kèm vệt quỹ đạo bắt mắt)."""
        # 1. Vẽ cầu thủ
        for track_id, data in projected_tracks.get("player", {}).items():
            x, y = map(int, np.round(data["position"]))
            team_id = data.get("team")
            if team_id == 1:
                fill, outline = (255, 255, 255), (0, 0, 0)
            elif team_id == 2:
                fill, outline = (0, 0, 0), (255, 255, 255)
            else:
                fill, outline = (128, 128, 128), (255, 255, 255)
            cv2.circle(minimap, (x, y), 7, outline, -1, cv2.LINE_AA)
            cv2.circle(minimap, (x, y), 5, fill, -1, cv2.LINE_AA)
            cv2.putText(minimap, str(track_id), (x + 6, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(minimap, str(track_id), (x + 6, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Vẽ trọng tài
        for track_id, data in projected_tracks.get("referee", {}).items():
            x, y = map(int, np.round(data["position"]))
            cv2.drawMarker(minimap, (x, y), (255, 80, 20), cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)

        # 3. Vẽ bóng kèm vệt chuyển động (Motion Trail)
        ball_tracks = projected_tracks.get("ball", {})
        if ball_tracks:
            # Chọn quả bóng có độ tin cậy cao nhất (tránh bóng giả / false positives)
            best_ball = max(
                ball_tracks.values(),
                key=lambda item: item.get("confidence", 1.0)
            )
            curr_pos = tuple(map(int, np.round(best_ball["position"])))
            self._ball_history.append(curr_pos)

            # Vẽ vệt đuôi quỹ đạo bóng (fading motion trail)
            if self.enable_ball_trail and len(self._ball_history) > 1:
                trail_pts = list(self._ball_history)
                for idx in range(len(trail_pts) - 1):
                    p_from = trail_pts[idx]
                    p_to = trail_pts[idx + 1]
                    progress = (idx + 1) / len(trail_pts)
                    trail_thickness = max(1, int(round(3 * progress)))
                    trail_color = (
                        int(20 + 30 * progress),
                        int(180 + 75 * progress),
                        int(230 + 25 * progress),
                    )
                    cv2.line(minimap, p_from, p_to, trail_color, trail_thickness, cv2.LINE_AA)

            # Vẽ bóng với 3 lớp nổi bật: Viền đen -> Viền trắng sáng -> Tâm cam/neon
            bx, by = curr_pos
            cv2.circle(minimap, (bx, by), 7, (0, 0, 0), -1, cv2.LINE_AA)       # Viền ngoài đen
            cv2.circle(minimap, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA) # Vòng đệm trắng
            cv2.circle(minimap, (bx, by), 4, (0, 215, 255), -1, cv2.LINE_AA)   # Tâm vàng cam rực rỡ
            cv2.circle(minimap, (bx, by), 1, (255, 255, 255), -1, cv2.LINE_AA) # Đốm sáng tâm bóng
        else:
            self._ball_history.clear()

        return minimap

    def render_minimap(
        self,
        tracks: Optional[dict] = None,
        frame_index: Optional[int] = None,
        homography: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.1,
        ransac_threshold: float = 14.0,
        min_inliers: int = 4,
        min_inlier_ratio: float = 0.15,
        keypoint_to_target: Optional[Mapping[int, int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        info = {"success": homography is not None}
        frame_has_saved_positions = False
        if tracks is not None and frame_index is not None:
            for key in ("player", "players", "referee", "ball"):
                frames = tracks.get(key, [])
                if frame_index < len(frames) and any(
                    "position_minimap" in item for item in frames[frame_index].values()
                ):
                    frame_has_saved_positions = True
                    break

        if frame_has_saved_positions:
            info = {"success": True, "precomputed_positions": True}
            if homography is None:
                stored_homographies = getattr(self, "_homographies_by_frame", [])
                homography = (
                    stored_homographies[frame_index]
                    if frame_index < len(stored_homographies)
                    else None
                )
            if homography is None:
                homography = np.eye(3, dtype=np.float64)
        elif tracks is not None and frame_index is not None and homography is None:
            keypoint_frames = tracks.get("keypoints", [])
            keypoints = (
                keypoint_frames[frame_index]
                if 0 <= frame_index < len(keypoint_frames)
                else {}
            )
            homography, info = self.calculate_homography(
                keypoints,
                confidence_threshold=confidence_threshold,
                ransac_threshold=ransac_threshold,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
                keypoint_to_target=keypoint_to_target,
            )

            if homography is not None:
                self._last_valid_homography = homography.copy()
                self._last_valid_homography_frame = frame_index
            elif (
                self._last_valid_homography is not None
                and self._last_valid_homography_frame is not None
                and frame_index - self._last_valid_homography_frame <= self.homography_hold_frames
            ):
                homography = self._last_valid_homography.copy()
                info = dict(info)
                info["usable"] = True
                info["success"] = False
                info["reused_previous_homography"] = True
                info["reason"] = "reusing last valid homography"
            else:
                search_end = min(len(keypoint_frames), frame_index + 31)
                for future_index in range(frame_index + 1, search_end):
                    candidate, candidate_info = self.calculate_homography(
                        keypoint_frames[future_index],
                        confidence_threshold=confidence_threshold,
                        ransac_threshold=ransac_threshold,
                        min_inliers=min_inliers,
                        min_inlier_ratio=min_inlier_ratio,
                        keypoint_to_target=keypoint_to_target,
                    )
                    if candidate is not None:
                        homography = candidate
                        self._last_valid_homography = candidate.copy()
                        self._last_valid_homography_frame = future_index
                        info = dict(candidate_info)
                        info["usable"] = True
                        info["success"] = False
                        info["reused_future_homography_frame"] = future_index
                        info["reason"] = f"using first available fit from frame {future_index}"
                        break

        minimap = self.draw_minimap()
        if tracks is not None and frame_index is not None and homography is not None:
            projected = self.project_tracks(tracks, frame_index, homography)
            minimap = self.draw_entities(minimap, projected)
        return minimap, homography, info

    def display_minimap(
        self,
        frame: np.ndarray,
        tracks: dict,
        frame_index: int,
        position: str = "bottom_center",
        keypoint_to_target: Optional[Mapping[int, int]] = None,
        confidence_threshold: float = 0.1,
        ransac_threshold: float = 14.0,
        min_inliers: int = 4,
        min_inlier_ratio: float = 0.15,
        alpha: float = 0.62,
    ) -> Tuple[np.ndarray, dict]:
        minimap, _, info = self.render_minimap(
            tracks=tracks,
            frame_index=frame_index,
            keypoint_to_target=keypoint_to_target,
            confidence_threshold=confidence_threshold,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
        )
        frame_h, frame_w = frame.shape[:2]
        max_w = max(1, min(self.width, frame_w - 16))
        max_h = max(1, min(self.height, frame_h - 16))
        scale = min(max_w / self.width, max_h / self.height)
        display_w = max(1, int(round(self.width * scale)))
        display_h = max(1, int(round(self.height * scale)))
        minimap = cv2.resize(minimap, (display_w, display_h), interpolation=cv2.INTER_AREA)

        if position == "bottom_center":
            x0 = (frame_w - display_w) // 2
            y0 = frame_h - display_h - 8
        elif position == "bottom_left":
            x0, y0 = 8, frame_h - display_h - 8
        elif position == "bottom_right":
            x0, y0 = frame_w - display_w - 8, frame_h - display_h - 8
        else:
            raise ValueError("position must be bottom_center, bottom_left, or bottom_right")

        x0 = max(0, min(x0, frame_w - display_w))
        y0 = max(0, min(y0, frame_h - display_h))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        roi = frame[y0:y0 + display_h, x0:x0 + display_w]
        frame[y0:y0 + display_h, x0:x0 + display_w] = cv2.addWeighted(
            minimap, alpha, roi, 1.0 - alpha, 0.0
        )
        border_color = (
            (30, 30, 30) if info.get("success")
            else (0, 180, 255) if info.get("usable")
            else (0, 0, 255)
        )
        cv2.rectangle(frame, (x0, y0), (x0 + display_w - 1, y0 + display_h - 1), border_color, 2)
        return frame, info
