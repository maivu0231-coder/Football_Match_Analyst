

from __future__ import annotations

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
    # Pass your own target_points/keypoint_to_target if model annotation order differs.
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
    ):
        if width <= 2 * margin or height <= 2 * margin:
            raise ValueError("Mini-map dimensions must be larger than its margins")
        self.width = int(width)
        self.height = int(height)
        self.margin = int(margin)
        self.background_color = tuple(map(int, background_color))
        self.line_color = tuple(map(int, line_color))
        scale = min(
            (self.width - 2 * self.margin) / self.PITCH_LENGTH,
            (self.height - 2 * self.margin) / self.PITCH_WIDTH,
        )
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

        # Correspondences inferred from the supplied labeled frames. Do not
        # assume that an unmapped model index equals the target index.
        self.keypoint_to_target = {
            13: 13,  # halfway line / far touchline
            14: 14,  # halfway line / upper center-circle point
            15: 17,  # halfway line / lower center-circle point
            16: 18,  # halfway line / near touchline
            17: 25,  # right penalty-area outer edge, top
            18: 26,  # right penalty-area edge / goal-area top
            19: 27,  # right penalty-area edge / goal-area bottom
            20: 28,  # right penalty-area outer edge, bottom
            21: 30,  # right goal-area inner edge, bottom
            22: 29,  # right goal-area inner edge, top
            23: 31,  # right penalty spot
            24: 19,  # right goal line, top
            25: 20,  # right goal line / penalty-area top
            26: 21,  # right goal line / goal-area top
            27: 22,  # right goal line / goal-area bottom
            28: 23,  # right goal line / penalty-area bottom
            30: 15,  # left point of center circle (low-confidence prediction)
            31: 16,  # right point of center circle
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

    def _meters_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        px = self.pitch_left + x / self.PITCH_LENGTH * self.pitch_pixel_length
        py = self.pitch_top + y / self.PITCH_WIDTH * self.pitch_pixel_width
        return float(px), float(py)

    def _pixel_to_meters(self, point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = map(float, point)
        pitch_x = (x - self.pitch_left) / self.pitch_pixel_length * self.PITCH_LENGTH
        pitch_y = (y - self.pitch_top) / self.pitch_pixel_width * self.PITCH_WIDTH
        return pitch_x, pitch_y

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
        """Create a clean mini-map image containing only pitch markings."""
        image = np.full(
            (self.height, self.width, 3), self.background_color, dtype=np.uint8
        )
        length, width = self.PITCH_LENGTH, self.PITCH_WIDTH
        half_width = width / 2.0
        pa_top = (width - self.PENALTY_AREA_WIDTH) / 2.0
        pa_bottom = width - pa_top
        ga_top = (width - self.GOAL_AREA_WIDTH) / 2.0
        ga_bottom = width - ga_top

        # Touchlines, goal lines, and halfway line.
        for start, end in (
            ((0.0, 0.0), (length, 0.0)),
            ((length, 0.0), (length, width)),
            ((length, width), (0.0, width)),
            ((0.0, width), (0.0, 0.0)),
            ((length / 2.0, 0.0), (length / 2.0, width)),
        ):
            self._draw_pitch_line(image, start, end)

        # Penalty areas, goal areas, and penalty spots at each end.
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
        """Estimate frame-pixel -> mini-map-pixel homography using RANSAC.

        ``image_keypoints`` is keyed by the model's stable keypoint index. Values
        may be ``{"position": [x,y], "confidence": score}`` or ``[x,y]``.
        Target coordinates are paired by ID, optionally translated through
        ``keypoint_to_target``. Returns the matrix and correspondence diagnostics
        (matched, inlier, and rejected IDs); returns ``None`` for an unusable fit.
        """
        id_mapping = dict(self.keypoint_to_target)
        if keypoint_to_target is not None:
            id_mapping.update({int(k): int(v) for k, v in keypoint_to_target.items()})

        source, destination, model_ids, target_ids = [], [], [], []
        for raw_id, value in image_keypoints.items():
            try:
                model_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            # Once an explicit map is configured, ignore model indices without
            # a known target correspondence instead of guessing identity.
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

        # Reject fits supported by a small or tightly clustered patch of the pitch.
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
            # Keep a mathematically usable fit so the map does not disappear
            # because the broadcast view exposes only a few landmarks.
            info["reason"] = "weak RANSAC fit; displaying with extra smoothing"
        self.last_homography_info = info
        return homography, info

    def transform_point(self, point, homography: np.ndarray) -> Optional[Tuple[float, float]]:
        if homography is None or point is None:
            return None
        point_array = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        projected = cv2.perspectiveTransform(point_array, homography.astype(np.float64))
        x, y = map(float, projected.reshape(2))
        if not np.isfinite([x, y]).all():
            return None
        # Bad projective fits often fling players far beyond the pitch.
        # Reject these positions instead of drawing them at arbitrary edges.
        if x < -8 or x > self.width + 8 or y < -8 or y > self.height + 8:
            return None
        return x, y

    @staticmethod
    def _track_image_position(track, object_type):
        # Homography maps the pitch plane (z=0). Use the ball box's bottom-center
        # as the closest image-space proxy to its ground projection; the center
        # drifts farther from the ground mapping when the ball is airborne.
        if object_type == "ball":
            bbox = track.get("bbox")
            if bbox is not None and len(bbox) >= 4:
                x1, y1, x2, y2 = map(float, bbox[:4])
                return ((x1 + x2) / 2.0, y2)
        position = track.get("position")
        if position is not None:
            return position
        bbox = track.get("bbox")
        if bbox is None or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = map(float, bbox[:4])
        return ((x1 + x2) / 2.0, y2)

    def project_tracks(
        self,
        tracks: dict,
        frame_index: int,
        homography: np.ndarray,
        smoothing: bool = True,
    ) -> dict:
        """Project player, referee, and ball positions to mini-map pixels."""
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
                    map_position = self.transform_point(image_position, homography)
                    used_stored_position = False
                if map_position is not None:
                    if smoothing and not used_stored_position:
                        map_position = self._smooth_track_position(
                            object_type, track_id, map_position, frame_index
                        )
                    projected[object_type][track_id] = {
                        "position": map_position,
                        "team_color": track.get("team_color"),
                        "team": track.get("team"),
                    }
        return projected

    def add_pitch_positions_to_tracks(
        self,
        tracks: dict,
        confidence_threshold: float = 0.25,
        ransac_threshold: float = 8.0,
        min_inliers: int = 5,
        min_inlier_ratio: float = 0.4,
    ) -> dict:
        """Store homography-projected mini-map pixels and pitch meters on tracks.

        Player/referee ground points use bbox bottom-center; ball uses bbox
        bottom-center as an approximate ground projection. These saved values
        are shared by mini-map rendering and distance/velocity measurements.
        """
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
            homography, fit_info = self.calculate_homography(
                keypoints,
                confidence_threshold=confidence_threshold,
                ransac_threshold=ransac_threshold,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
            )
            # calculate_homography can return a geometrically possible but
            # weak fit for visualization. Do not use that fit for metric
            # coordinates; it is a common source of speed/distance spikes.
            if homography is not None and fit_info.get("success", False):
                last_homography = homography
                last_valid_frame = frame_index
            else:
                homography = None
            if homography is None and (
                last_homography is not None
                and last_valid_frame is not None
                and frame_index - last_valid_frame <= self.homography_hold_frames
            ):
                homography = last_homography
            homographies[frame_index] = homography

        # Seed the opening frames from the first usable calibration rather than
        # leaving their coordinates empty.
        first_valid = next(
            (index for index, homography in enumerate(homographies) if homography is not None),
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
        for frame_index, homography in enumerate(homographies):
            if homography is None:
                continue
            projected = self.project_tracks(
                tracks, frame_index, homography, smoothing=True
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

        # Fill frames where this track exists but that frame had no usable H.
        # Both saved coordinate forms are interpolated together so the mini-map
        # and measurement code continue to use the exact same ground position.
        for object_type, key in object_keys.items():
            object_frames = tracks.get(key, [])
            observations = {}
            for frame_index, frame_tracks in enumerate(object_frames):
                for track_id, track in frame_tracks.items():
                    observations.setdefault(track_id, []).append((frame_index, track))

            for records in observations.values():
                known = [
                    (frame_index, np.asarray(track["position_minimap"], dtype=np.float64))
                    for frame_index, track in records
                    if track.get("position_minimap") is not None
                ]
                if not known:
                    continue
                known_frames = np.asarray([frame for frame, _ in known], dtype=np.float64)
                known_positions = np.asarray([position for _, position in known], dtype=np.float64)
                for frame_index, track in records:
                    if track.get("position_minimap") is not None:
                        continue
                    interpolated = [
                        float(np.interp(frame_index, known_frames, known_positions[:, axis]))
                        for axis in range(2)
                    ]
                    track["position_minimap"] = interpolated
                    track["position_pitch"] = list(self._pixel_to_meters(interpolated))

        # Retain the per-frame transforms for rendering diagnostics, then clear
        # the streaming fallback state so frame zero cannot reuse the last frame's H.
        self._homographies_by_frame = homographies
        self._last_valid_homography = None
        self._last_valid_homography_frame = None
        return tracks

    def _smooth_track_position(self, object_type, track_id, position, frame_index):
        """Stabilize projected tracks without making the ball trail the play."""
        key = (object_type, track_id)
        current = np.asarray(position, dtype=np.float32)
        previous_state = self._smoothed_positions.get(key)
        if previous_state is None:
            self._smoothed_positions[key] = (frame_index, current)
            return tuple(map(float, current))

        previous_frame, previous = previous_state
        elapsed = max(1, int(frame_index) - int(previous_frame))
        if elapsed > 5:
            self._smoothed_positions[key] = (frame_index, current)
            if object_type == "ball":
                self._ball_smoothed_velocity[key] = np.zeros(2, dtype=np.float32)
            return tuple(map(float, current))

        if object_type == "ball":
            velocity = self._ball_smoothed_velocity.get(
                key, np.zeros(2, dtype=np.float32)
            )
            predicted = previous + velocity * elapsed
            residual = current - predicted
            residual_size = float(np.linalg.norm(residual))

            if residual_size <= 2.5:
                # Ignore tiny frame-to-frame projection wobble.
                smoothed = predicted
                velocity *= 0.85
            elif residual_size <= 16.0:
                # Damp moderate jitter while allowing smooth rolling movement.
                smoothed = predicted + 0.22 * residual
                observed_velocity = (smoothed - previous) / elapsed
                velocity = 0.6 * velocity + 0.4 * observed_velocity
            else:
                # A large residual is likely a real kick. Follow it quickly,
                # with a per-frame cap to suppress a one-frame false detection.
                correction = residual.copy()
                max_correction = 34.0 * elapsed
                if residual_size > max_correction:
                    correction *= max_correction / residual_size
                direction_agreement = float(np.dot(velocity, residual))
                gain = 0.76 if direction_agreement >= 0 else 0.62
                smoothed = predicted + gain * correction
                observed_velocity = (smoothed - previous) / elapsed
                velocity = 0.45 * velocity + 0.55 * observed_velocity

            max_speed = 38.0 * elapsed
            actual_step = smoothed - previous
            actual_step_size = float(np.linalg.norm(actual_step))
            if actual_step_size > max_speed:
                smoothed = previous + actual_step * (max_speed / actual_step_size)
            self._ball_smoothed_velocity[key] = velocity.astype(np.float32)
        else:
            # Keep player and referee points steady against small homography noise.
            alpha = 0.18
            delta = current - previous
            distance = float(np.linalg.norm(delta))
            max_step = 5.0 * elapsed
            if distance > max_step:
                current = previous + delta * (max_step / distance)
            smoothed = previous * (1.0 - alpha) + current * alpha

        self._smoothed_positions[key] = (frame_index, smoothed)
        return tuple(map(float, smoothed))

    def draw_entities(self, minimap: np.ndarray, projected_tracks: dict) -> np.ndarray:
        """Draw projected players, referees, and ball on an existing mini-map."""
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

        for track_id, data in projected_tracks.get("referee", {}).items():
            x, y = map(int, np.round(data["position"]))
            cv2.drawMarker(minimap, (x, y), (255, 80, 20), cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)

        for _, data in projected_tracks.get("ball", {}).items():
            x, y = map(int, np.round(data["position"]))
            cv2.circle(minimap, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(minimap, (x, y), 6, (0, 0, 0), 1, cv2.LINE_AA)
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
        """Render pitch and entities; estimate homography from this frame if needed."""
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
                # Tracks are precomputed for the whole video. Reuse the last
                # usable transform on frames with missing/weak landmarks.
                homography = self._last_valid_homography.copy()
                info = dict(info)
                info["usable"] = True
                info["success"] = False
                info["reused_previous_homography"] = True
                info["reason"] = "reusing last valid homography"
            else:
                # At the start of a video there is no previous transform yet.
                # Look ahead briefly for the first frame with enough landmarks.
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
        """Overlay the frame's mini-map at the bottom-center of the video frame."""
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
