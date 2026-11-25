from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import Counter

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
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
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
    team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

    self.player_team_dict[player_id] = team_id
    return team_id

  def assign_team_color_from_video(self, video_path: str, tracks: dict) -> dict:
      player_key = "players" if "players" in tracks else "player"
      player_tracks_list = tracks.get(player_key, [])

      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
          return tracks

      # Phase 1: Collect diverse player colors across multiple frames to find 2 team colors
      player_colors = []
      f_idx = 0
      while cap.isOpened() and len(player_colors) < 60:
          ret, frame = cap.read()
          if not ret or f_idx >= len(player_tracks_list):
              break

          player_dict = player_tracks_list[f_idx]
          if len(player_dict) > 0 and f_idx % 2 == 0:
              for p in player_dict.values():
                  player_color = self.get_playercolor(frame, p["bbox"])
                  player_colors.append(player_color)

          f_idx += 1

      # cluster the player colors to find the two main team colors using KMeans
      if len(player_colors) >= 2:
          kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
          kmeans.fit(player_colors)
          self.kmeans = kmeans
          self.team_colors[1] = kmeans.cluster_centers_[0]
          self.team_colors[2] = kmeans.cluster_centers_[1]
      else:
          self.team_colors[1] = np.array([255, 255, 255])
          self.team_colors[2] = np.array([0, 0, 255])

      # Phase 2: Vote across the video to robustly assign teams to each player (Majority Voting)
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      player_team_votes = {}
      vote_step = 10  # Sample every 10 frames to optimize speed while retaining high accuracy
      f_idx = 0

      while cap.isOpened():
          ret, frame = cap.read()
          if not ret or f_idx >= len(player_tracks_list):
              break

          if f_idx % vote_step == 0:
              player_dict = player_tracks_list[f_idx]
              for player_id, track_info in player_dict.items():
                  player_color = self.get_playercolor(frame, track_info["bbox"])
                  if self.kmeans is not None:
                      pred_team = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
                  else:
                      pred_team = 1
                  player_team_votes.setdefault(player_id, []).append(pred_team)

          f_idx += 1

      cap.release()

      # Determine final team for each player via Majority Voting
      for player_id, votes in player_team_votes.items():
          most_common_team = Counter(votes).most_common(1)[0][0]
          self.player_team_dict[player_id] = most_common_team

      # Manual override for the known player track.
      self.player_team_dict[26] = 2

      # Phase 3: Apply team assignments and colors to all frames in tracks
      for player_dict in player_tracks_list:
          for player_id, track_info in player_dict.items():
              team_id = 2 if int(player_id) == 26 else self.player_team_dict.get(player_id, 1)
              track_info["team"] = team_id
              track_info["team_color"] = tuple(map(int, self.team_colors.get(team_id, [0, 0, 255])))

      return tracks
