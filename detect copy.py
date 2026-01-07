import csv
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter
from tqdm import tqdm


WINDOW_SIZE = 100
STABLE_COUNT = 4
ASSOCIATION_THRESHOLD = 25.0
MAX_MISSED_TIME = 10.0


class RadarTrack:
    def __init__(self, track_id, pulse):
        self.id = track_id
        self.confidence = 1
        self.last_update = pulse["toa"]

        self.pw_vals = deque(maxlen=WINDOW_SIZE)
        self.pri_vals = deque(maxlen=WINDOW_SIZE)

        self.pw_vals.append(pulse["pulse_width"])
        self.pri_vals.append(pulse["pri"])

        self.min_doa = pulse["doa"]
        self.max_doa = pulse["doa"]
        self.min_freq = pulse["frequency"]
        self.max_freq = pulse["frequency"]

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([pulse["doa"], pulse["frequency"], 0, 0])
        self.kf.H = np.array([[1,0,0,0],
                              [0,1,0,0]])
        self.kf.P *= 100
        self.kf.R *= 5
        self.kf.Q *= 0.01

    def predict(self, current_time):
        dt = max(current_time - self.last_update, 0.1)
        self.kf.F = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]
        ])
        self.kf.predict()

    def update(self, pulse):
        self.kf.update([pulse["doa"], pulse["frequency"]])

        self.pw_vals.append(pulse["pulse_width"])
        self.pri_vals.append(pulse["pri"])

        self.min_doa = min(self.min_doa, pulse["doa"])
        self.max_doa = max(self.max_doa, pulse["doa"])
        self.min_freq = min(self.min_freq, pulse["frequency"])
        self.max_freq = max(self.max_freq, pulse["frequency"])

        self.last_update = pulse["toa"]
        self.confidence += 1

    def distance(self, pulse):
        doa_pred, freq_pred = self.kf.x[0], self.kf.x[1]

        return (
            abs(doa_pred - pulse["doa"]) * 0.5 +
            abs(freq_pred - pulse["frequency"]) * 0.01 +
            abs(np.mean(self.pw_vals) - pulse["pulse_width"]) * 1.0 +
            abs(np.mean(self.pri_vals) - pulse["pri"]) * 0.3
        )


class RadarTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def process_pulse(self, pulse):
        best_track = None
        best_cost = float("inf")

        for track in self.tracks.values():
            track.predict(pulse["toa"])
            cost = track.distance(pulse)
            if cost < best_cost:
                best_cost = cost
                best_track = track

        if best_cost < ASSOCIATION_THRESHOLD:
            best_track.update(pulse)
        else:
            self.tracks[self.next_id] = RadarTrack(self.next_id, pulse)
            self.next_id += 1

        self._delete_old_tracks(pulse["toa"])

    def _delete_old_tracks(self, current_time):
        to_remove = [
            tid for tid, trk in self.tracks.items()
            if current_time - trk.last_update > MAX_MISSED_TIME
        ]
        for tid in to_remove:
            del self.tracks[tid]

    def confirmed_tracks(self):
        return [
            t for t in self.tracks.values()
            if t.confidence >= STABLE_COUNT
        ]


tracker = RadarTracker()

with open("radar_pulses.csv", "r") as f:
    total_pulses = sum(1 for _ in f) - 1

with open("radar_pulses.csv", "r") as file:
    reader = csv.DictReader(file)

    for idx, row in enumerate(tqdm(reader, total=total_pulses, desc="Processing Pulses"), start=1):
        pulse = {
            "pulse_width": float(row["pulse_width"]),
            "doa": float(row["doa"]),
            "pri": float(row["pri"]),
            "frequency": float(row["frequency"]),
            "toa": idx * 0.1
        }
        tracker.process_pulse(pulse)


print("\n FINAL SHIP IDENTIFICATION RESULT \n")

for track in tracker.confirmed_tracks():
    ship_list = [
        f"minDOA = {track.min_doa:.2f}",
        f"maxDOA = {track.max_doa:.2f}",
        f"minFreq = {track.min_freq:.2f}",
        f"maxFreq = {track.max_freq:.2f}",
        f"avgPW = {np.mean(track.pw_vals):.2f}",
        f"Pulses = {track.confidence}"
    ]
    print(f"Ship {track.id + 1}: {ship_list}")

print("\nFINAL SHIP COUNT:", len(tracker.confirmed_tracks()))
