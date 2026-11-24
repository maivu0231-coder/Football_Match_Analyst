import cv2
import numpy as np


class ViewTransformer:

    def __init__(self):
        court_width = 68.0  #the width of a standard football pitch in meters
        court_length = 23.32  #4 cục

        self.pixel_vertices = np.array(
            [
                [750,843 ],  # Điểm dưới cùng bên trái
                [870, 310],  # Điểm trên cùng bên trái
                [1382, 306],  # Điểm trên cùng bên phải
                [1630, 845],  # Điểm dưới cùng bên phải
            ],
            dtype=np.float32,
        )
        self.target_vertices = np.array(
            [
                [0, court_width],  # Dưới trái
                [0, 0],  # Trên trái
                [court_length, 0],  # Trên phải
                [court_length, court_width],  # Dưới phải
            ],
            dtype=np.float32,
        )

        #compute homography matrix for perspective transformation
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        if point is None:
            return None

        p = (int(point[0]), int(point[1]))

        reshaped_point = np.array([p], dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transformer
    )
        return transformed_point.reshape(-1, 2)[0].tolist()

    def add_transformed_position_to_tracks(self, tracks: dict)-> dict:
        #add a new key "position_transformed" to each track's info dictionary, which contains the transformed position in meters
        for obj, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # Lấy tọa độ chân đã bù trừ camera
                    position = track_info.get(
                        "position_adjusted", track_info.get("position")
                    )

                    if position is not None:
                        position_transformed = self.transform_point(position)
                        track_info["position_transformed"] = (
                            position_transformed
                        )
        return tracks