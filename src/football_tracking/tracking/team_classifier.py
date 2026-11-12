from __future__ import annotations
from collections import defaultdict
import cv2
import numpy as np
from football_tracking.tracking.tracker import TrackedObject

class TeamClassifier:
    def __init__(self, number_of_teams: int = 2, minimum_samples: int = 6, samples_per_player: int = 30) -> None:
        if number_of_teams != 2:
            raise ValueError("El clasificador soporta exactamente 2 equipos")

        if minimum_samples < 2:
            raise ValueError("Los samples al menos deben ser 2 como mínimo")

        if samples_per_player < 1:
            raise ValueError("Los samples de jugadores deben ser positivos")

        self.number_of_teams = number_of_teams
        self.minimum_samples = minimum_samples
        self.samples_per_player = samples_per_player
        self._player_samples: dict[int, list[np.ndarray]] = defaultdict(list)
        self._team_assignments: dict[int, int] = {}
        self._team_centers: np.ndarray | None = None

    def _extract_jersey_color(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, int(x1)))
        x2 = max(0, min(width, int(x2)))
        y1 = max(0, min(height, int(y1)))
        y2 = max(0, min(height, int(y2)))
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < 5 or box_height < 10:
            return None

        crop = frame[y1 + int(box_height * 0.15): y1 + int(box_height * 0.55), x1 + int(box_width * 0.20): x1 + int(box_width * 0.80)]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        hue = pixels[:, 0]
        saturation = pixels[:, 1]
        value = pixels[:, 2]
        mask = (((saturation < 40) | (hue < 30) | (hue > 95)) & (value > 35))
        valid_pixels = pixels[mask]

        if len(valid_pixels) < 5:
            return None

        return np.median(valid_pixels, axis=0).astype(np.float32)

    def _update_samples(self, frame: np.ndarray, tracked_objects: list[TrackedObject]) -> None:
        for tracked_object in tracked_objects:
            if tracked_object.class_name != "player":
                continue

            track_id = tracked_object.track_id

            if track_id in self._team_assignments:
                continue

            samples = self._player_samples[track_id]

            if len(samples) >= self.samples_per_player:
                continue

            color = self._extract_jersey_color(frame, tracked_object.bbox)

            if color is not None:
                samples.append(color)

    def _fit_clusters(self) -> None:
        player_colors = []
        player_ids = []

        for track_id, samples in self._player_samples.items():
            if len(samples) < 2:
                continue

            player_colors.append(np.mean(samples, axis=0))
            player_ids.append(track_id)

        if len(player_colors) < self.minimum_samples:
            return

        data = np.asarray(player_colors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(data, self.number_of_teams, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        self._team_centers = centers

        for track_id, label in zip(player_ids, labels.flatten()):
            self._team_assignments[track_id] = int(label)

    def _assign_new_players(self) -> None:
        if self._team_centers is None:
            return

        for track_id, samples in self._player_samples.items():
            if track_id in self._team_assignments or len(samples) < 2:
                continue

            player_color = np.mean(samples, axis=0)
            distances = np.linalg.norm(self._team_centers - player_color, axis=1)
            self._team_assignments[track_id] = int(np.argmin(distances))

    def classify(self, frame: np.ndarray, tracked_objects: list[TrackedObject]) -> dict[int, int]:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3).")

        self._update_samples(frame, tracked_objects)

        if self._team_centers is None:
            self._fit_clusters()
        self._assign_new_players()
        return self._team_assignments.copy()

    def reset(self) -> None:
        self._player_samples.clear()
        self._team_assignments.clear()
        self._team_centers = None
