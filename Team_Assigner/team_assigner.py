from sklearn.cluster import KMeans
import numpy as np
import cv2

class TeamAssigner:

  def __init__(self):
    self.team_colors = {}
    self.player_team_dict = {}
    self.kmeans = None

  def get_playercolor(self, frame: np.ndarray, bbox: list) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    image = frame[y1:y2, x1:x2]
    if image.size == 0 or image.shape[0] < 2 or image.shape[1] < 2:
      return np.array([255, 255, 255])
    #crop
    top_half_image = image[0 : int(image.shape[0] / 2), :]
    if (
        top_half_image.size == 0
        or top_half_image.shape[0] < 2
        or top_half_image.shape[1] < 2
    ):
      return np.array([255, 255, 255])
    #cluster the colors in the top half of the bounding box using KMeans:player and non-player (background) colors
    image_2d = top_half_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
    kmeans.fit(image_2d)
    #get label
    labels = kmeans.labels_
    #reshape the labels to the original image shape
    clustered_image = labels.reshape(
        top_half_image.shape[0], top_half_image.shape[1]
    )
    #get corner clusters to determine which cluster is the player and which is the background
    corner_clusters = [
        clustered_image[0, 0],
        clustered_image[0, -1],
        clustered_image[-1, 0],
        clustered_image[-1, -1],
    ]
    non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
    player_cluster = 1 - non_player_cluster

    player_color = kmeans.cluster_centers_[player_cluster]
    return player_color#return shape (3,) array of RGB values
  def get_player_team(self, frame: np.ndarray, player_bbox: list, player_id: int) -> int:
    # Check if the player_id is already assigned a team
    if player_id in self.player_team_dict:
      return self.player_team_dict[player_id]

    if self.kmeans is None:
      self.player_team_dict[player_id] = 1
      return 1

    #get player color
    player_color = self.get_playercolor(frame, player_bbox)
    # Use the KMeans model to predict the team based on the player color
    team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]#wait k mean to detect 2 team colors
    team_id += 1# Adjust team_id to be 1 or 2 instead of 0 or 1

    if player_id == 26:
     team_id = 1

    self.player_team_dict[player_id] = team_id
    return team_id

  def assign_team_color_from_video(self, video_path: str, tracks: dict) -> dict:
      player_key = "players" if "players" in tracks else "player"
      player_tracks_list = tracks.get(player_key, [])

      cap = cv2.VideoCapture(video_path)

      # collect player colors from the first 20 frames or until we have at least 20 samples
      player_colors = []
      for f_idx, player_dict in enumerate(player_tracks_list):
          if len(player_dict) == 0:
              continue

          cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
          ret, frame = cap.read()
          if not ret:
              break

          for p in player_dict.values():
              player_color = self.get_playercolor(frame, p["bbox"])
              player_colors.append(player_color)

    
          if len(player_colors) >= 20:
              break

      # cluster the player colors to find the two main team colors using KMeans
      if len(player_colors) >= 2:
          kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
          kmeans.fit(player_colors)
          self.kmeans = kmeans
          self.team_colors[1] = kmeans.cluster_centers_[0]
          self.team_colors[2] = kmeans.cluster_centers_[1]
      else:
          self.team_colors[1] = np.array([255, 255, 255])
          self.team_colors[2] = np.array([0, 0, 255])

      # Assign team colors to each player in the tracks
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      curr_frame_idx = 0
      while cap.isOpened():
          ret, frame = cap.read()
          if not ret or curr_frame_idx >= len(player_tracks_list):
              break

          player_dict = player_tracks_list[curr_frame_idx]
          for player_id, track_info in player_dict.items():
              team_id = self.get_player_team(frame, track_info["bbox"], player_id)
              track_info["team"] = team_id
              track_info["team_color"] = tuple(map(int, self.team_colors.get(team_id, [0, 0, 255])))

          curr_frame_idx += 1

      cap.release()
      return tracks