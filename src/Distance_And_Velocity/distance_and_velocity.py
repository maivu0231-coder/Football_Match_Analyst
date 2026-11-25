import cv2
import numpy as np


class Measurement:
    def __init__(
        self,
        frame_rate: int = 24,
        frame_window: int = 5,
        pixel_to_meter_ratio: float = 0.04,
        position_smoothing_window: int = 5,
        max_speed_mps: float = 12.0,
        min_movement_meters: float = 0.03,
    ):
        self.frame_rate = frame_rate
        self.frame_window = max(1, int(frame_window))
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.position_smoothing_window = max(1, int(position_smoothing_window))
        if self.position_smoothing_window % 2 == 0:
            self.position_smoothing_window += 1
        self.max_speed_mps = float(max_speed_mps)
        self.min_movement_meters = float(min_movement_meters)

    def add_distance_and_velocity(self, tracks: dict) -> dict:
        for object_name, object_tracks in tracks.items():
            if object_name not in ["player", "players"]:
                continue
            observations = {}
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    track_info.pop("speed", None)
                    track_info.pop("velocity", None)
                    track_info.pop("distance", None)
                    # Prefer the per-frame pitch homography. If it is absent,
                    # use the legacy perspective transform of position_adjusted
                    # (already stored in meters by ViewTransformer).
                    position = track_info.get("position_pitch")
                    if position is None:
                        position = track_info.get("position_transformed")
                    if position is None:
                        continue
                    point = np.asarray(position, dtype=np.float64).reshape(-1)
                    if point.size < 2 or not np.isfinite(point[:2]).all():
                        continue
                    observations.setdefault(track_id, []).append(
                        (frame_num, frame_tracks[track_id], point[:2])
                    )

            for records in observations.values():
                records.sort(key=lambda item: item[0])
                raw_positions = np.asarray(
                    [item[2] for item in records], dtype=np.float64
                )

                # A short centered median removes one-frame homography/box jitter
                # while preserving sustained player movement.
                radius = self.position_smoothing_window // 2
                smoothed_positions = np.empty_like(raw_positions)
                for position_index in range(len(raw_positions)):
                    start = max(0, position_index - radius)
                    end = min(len(raw_positions), position_index + radius + 1)
                    smoothed_positions[position_index] = np.median(
                        raw_positions[start:end], axis=0
                    )

                cumulative_distance = 0.0
                speed_values = []
                for index, (frame_num, track_info, _) in enumerate(records):
                    position = smoothed_positions[index]
                    if index > 0:
                        previous_frame = records[index - 1][0]
                        previous_position = smoothed_positions[index - 1]
                        elapsed_seconds = (frame_num - previous_frame) / self.frame_rate
                        step_distance = float(np.linalg.norm(position - previous_position))
                        plausible_step = (
                            elapsed_seconds > 0
                            and step_distance / elapsed_seconds <= self.max_speed_mps
                        )
                        if plausible_step:
                            if step_distance >= self.min_movement_meters:
                                cumulative_distance += step_distance

                    # Measure over the configured time span, not the next
                    # available observation (which was effectively one frame).
                    target_index = next(
                        (
                            candidate_index
                            for candidate_index in range(index + 1, len(records))
                            if records[candidate_index][0] - frame_num >= self.frame_window
                        ),
                        None,
                    )
                    origin_index = index
                    if target_index is None and index > 0:
                        origin_index = max(
                            0,
                            next(
                                (
                                    candidate_index
                                    for candidate_index in range(index - 1, -1, -1)
                                    if frame_num - records[candidate_index][0]
                                    >= self.frame_window
                                ),
                                0,
                            ),
                        )
                        target_index = index

                    if target_index is not None:
                        origin_frame = records[origin_index][0]
                        target_frame = records[target_index][0]
                        elapsed_seconds = (target_frame - origin_frame) / self.frame_rate
                        distance_meters = float(
                            np.linalg.norm(
                                smoothed_positions[target_index]
                                - smoothed_positions[origin_index]
                            )
                        )
                        measured_speed = distance_meters / elapsed_seconds if elapsed_seconds > 0 else 0.0
                        speed_km_per_hour = (
                            measured_speed * 3.6
                            if measured_speed <= self.max_speed_mps
                            else 0.0
                        )
                    else:
                        speed_km_per_hour = 0.0

                    track_info["distance"] = cumulative_distance
                    speed_values.append(speed_km_per_hour)

                # Smooth the displayed speed itself so a valid but slightly
                # changing frame homography does not make the label flicker.
                speed_radius = 2
                for index, (_, track_info, _) in enumerate(records):
                    start = max(0, index - speed_radius)
                    end = min(len(speed_values), index + speed_radius + 1)
                    stable_speed = float(np.median(speed_values[start:end]))
                    track_info["speed"] = stable_speed
                    track_info["velocity"] = stable_speed
        return tracks

    add_distance_and_velocity_to_tracks = add_distance_and_velocity
    add_speed_and_distance_to_tracks = add_distance_and_velocity

    def draw_distance_and_velocity(self, frames, tracks=None, frame_num: int = None)-> any:
        if isinstance(frames, list):
            #draw annotations on each frame in the list and return the list
            for f_idx, frame in enumerate(frames):
                self._draw_frame(frame, tracks, f_idx)
            return frames
        elif hasattr(frames, "__next__") or (hasattr(frames, "__iter__") and not isinstance(frames, np.ndarray)):
            #using a generator, yield annotated frames one by one
            def generator():
                for f_idx, frame in enumerate(frames):
                    self._draw_frame(frame, tracks, f_idx)
                    yield frame
            return generator()
        else:
            #draw annotations on a single frame and return the frame
            return self._draw_frame(frames, tracks, frame_num)
    draw_speed_and_distance = draw_distance_and_velocity

    def _draw_frame(self, frame: np.ndarray, tracks: any, frame_num: int = None) -> np.ndarray:
        if frame is None or tracks is None:
            return frame

        player_tracks = {}
        if isinstance(tracks, dict):
            if "player" in tracks or "players" in tracks:
                player_list = tracks.get("player", tracks.get("players", []))
                if frame_num is not None and 0 <= frame_num < len(player_list):
                    player_tracks = player_list[frame_num]
                elif frame_num is None and len(player_list) > 0:
                    player_tracks = player_list[0]
            else:
            
                player_tracks = tracks
        elif isinstance(tracks, list):
            if frame_num is not None and 0 <= frame_num < len(tracks):
                player_tracks = tracks[frame_num]

        for track_id, track_info in player_tracks.items():
            if not isinstance(track_info, dict):
                continue

            speed = track_info.get("speed", track_info.get("velocity"))
            distance = track_info.get("distance")

            if speed is None or distance is None:
                continue

            # get the position of the player to place the text annotations
            bbox = track_info.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)
                width = x2 - x1
                axes_y = int(width / 6)  
            elif "position" in track_info:
                center_x, center_y = [int(v) for v in track_info["position"]]
                axes_y = 10
            else:
                continue

            speed_text = f"{speed:.2f} km/h"
            distance_text = f"{distance:.2f} m"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            color = (0, 0, 0)

            (w_speed, _), _ = cv2.getTextSize(speed_text, font, font_scale, thickness)
            (w_dist, _), _ = cv2.getTextSize(distance_text, font, font_scale, thickness)

         
            speed_x = int(center_x - w_speed / 2)
            speed_y = int(center_y + axes_y + 35)

            dist_x = int(center_x - w_dist / 2)
            dist_y = int(speed_y + 20)

            cv2.putText(frame, speed_text, (speed_x, speed_y), font, font_scale, color, thickness)
            cv2.putText(frame, distance_text, (dist_x, dist_y), font, font_scale, color, thickness)

        return frame
