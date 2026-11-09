import cv2
import numpy as np


class Measurement:
    def __init__(
        self,
        frame_rate: int = 24,
        frame_window: int = 5,
        pixel_to_meter_ratio: float = 0.04,
    ):
        self.frame_rate = frame_rate
        self.frame_window = frame_window
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
    def add_distance_and_velocity(self, tracks: dict) -> dict:
        total_distance = {}
        for object_name, object_tracks in tracks.items():
            if object_name not in ["player", "players"]:
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                for track_id, track_info in object_tracks[frame_num].items():
                    target_frame = last_frame
                    if track_id not in object_tracks[target_frame]:
                        found = False
                        for f in range(last_frame - 1, frame_num, -1):
                            if track_id in object_tracks[f]:
                                target_frame = f
                                found = True
                                break
                        if not found:
                            continue
                    delta_frame = target_frame - frame_num
                    if delta_frame <= 0:
                        continue
                    start_position = track_info.get("position_transformed")
                    end_position = object_tracks[target_frame][track_id].get("position_transformed")
                    if start_position is None or end_position is None:
                        continue
                    # 1. Khoảng cách bằng PIXEL
                    distance_covered_pixels = float(
                        np.linalg.norm(
                            np.array(start_position) - np.array(end_position)
                        )
                    )
                    #compute time
                    time_seconds = delta_frame / self.frame_rate
                    #compute speed
                    speed_m_per_s = distance_covered_pixels/time_seconds
                    #convert speed to km/h
                    speed_km_per_hour = speed_m_per_s * 3.6

                    # compute distance in meters
                    if track_id not in total_distance:
                        total_distance[track_id] = 0.0
                    total_distance[track_id] += distance_covered_pixels 
                    # add distance and speed to the track info for all frames in the window
                    end_range = (
                        target_frame + 1
                        if target_frame == number_of_frames - 1
                        else target_frame
                    )
                    for f_num in range(frame_num, end_range):
                        if track_id in object_tracks[f_num]:
                            object_tracks[f_num][track_id]["distance"] = (
                                total_distance[track_id]
                            )
                            object_tracks[f_num][track_id]["speed"] = (
                                speed_km_per_hour
                            )
                            object_tracks[f_num][track_id]["velocity"] = (
                                speed_km_per_hour
                            )
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
