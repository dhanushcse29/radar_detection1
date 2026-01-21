
# # radar_client_dynamic_pri.py
# import socket
# import pickle
# import time
# import numpy as np
# from sklearn.cluster import DBSCAN

# # ---------------- CONFIG ----------------
# HOST = "127.0.0.1"
# PORT = 65432

# WINDOW_SIZE = 5000            # max buffer size
# CLUSTER_BATCH_SIZE = 1500     # cluster every 1500 pulses
# MIN_SAMPLES = 15              # min samples for clustering
# SOFT_MIN_SAMPLES = 5          # relaxed clustering at end-of-stream
# DOA_GATE = 5.0
# FREQ_GATE = 20.0
# # ---------------------------------------

# # ---------------- STORAGE ----------------
# pulse_buffer = []
# pulse_counter_since_cluster = 0
# last_toa = None  # For dynamic PRI calculation
# emitters = {}
# emitter_id_counter = 0
# total_pulses_received = 0  # Track total received pulses
# # -----------------------------------------

# # --------- EMITTER FUNCTIONS ---------
# def associate_emitter(stats):
#     for eid, e in emitters.items():
#         if abs(stats["meanDOA"] - e["meanDOA"]) < DOA_GATE and \
#            abs(stats["meanFreq"] - e["meanFreq"]) < FREQ_GATE:
#             return eid
#     return None

# def update_emitter(eid, stats):
#     e = emitters[eid]
#     e["meanDOA"]  = 0.7 * e["meanDOA"]  + 0.3 * stats["meanDOA"]
#     e["meanFreq"] = 0.7 * e["meanFreq"] + 0.3 * stats["meanFreq"]
#     e["minDOA"]   = min(e["minDOA"], stats["minDOA"])
#     e["maxDOA"]   = max(e["maxDOA"], stats["maxDOA"])
#     e["minFreq"]  = min(e["minFreq"], stats["minFreq"])
#     e["maxFreq"]  = max(e["maxFreq"], stats["maxFreq"])
#     e["pulseCount"] = stats["pulseCount"]  # Replace with current cluster count

# def create_emitter(stats):
#     global emitter_id_counter
#     emitters[emitter_id_counter] = {
#         "meanDOA": stats["meanDOA"],
#         "meanFreq": stats["meanFreq"],
#         "minDOA": stats["minDOA"],
#         "maxDOA": stats["maxDOA"],
#         "minFreq": stats["minFreq"],
#         "maxFreq": stats["maxFreq"],
#         "pulseCount": stats["pulseCount"]
#     }
#     emitter_id_counter += 1

# # --------- PROCESS BUFFER ---------
# def process_buffer(buffer_list, min_samples):
#     if len(buffer_list) < min_samples:
#         return

#     # Extract DOA and frequency for clustering
#     data = np.array([(p["doa"], p["frequency"]) for p in buffer_list], dtype=np.float32)

#     # Normalize features
#     means = data.mean(axis=0)
#     stds = data.std(axis=0)
#     stds[stds == 0] = 1
#     X_scaled = (data - means) / stds

#     # Use DBSCAN for clustering
#     eps = 0.3
#     labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_scaled)

#     # Process clusters
#     unique_labels = np.unique(labels)
#     for cid in unique_labels:
#         if cid == -1:
#             continue
#         mask = labels == cid
#         cluster_pulses = [buffer_list[i] for i in np.where(mask)[0]]

#         doas = np.array([p["doa"] for p in cluster_pulses])
#         freqs = np.array([p["frequency"] for p in cluster_pulses])

#         stats = {
#             "meanDOA": float(doas.mean()),
#             "meanFreq": float(freqs.mean()),
#             "minDOA": float(doas.min()),
#             "maxDOA": float(doas.max()),
#             "minFreq": float(freqs.min()),
#             "maxFreq": float(freqs.max()),
#             "pulseCount": len(cluster_pulses)
#         }

#         eid = associate_emitter(stats)
#         if eid is None:
#             create_emitter(stats)
#         else:
#             update_emitter(eid, stats)

# # ---------------- MAIN CLIENT ----------------
# if __name__ == "__main__":
#     start_time = time.perf_counter()
#     window_counter = 0

#     print("Connecting to server...")

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
#         client.connect((HOST, PORT))
#         print("Connected to server")

#         while True:
#             length_bytes = client.recv(8)
#             if not length_bytes:
#                 break

#             data_length = int.from_bytes(length_bytes, "big")
#             payload = b""
#             while len(payload) < data_length:
#                 packet = client.recv(data_length - len(payload))
#                 if not packet:
#                     break
#                 payload += packet
#             if not payload:
#                 break

#             window_counter += 1
#             raw_window = pickle.loads(payload)
#             print(f"t{window_counter} → pulses arrived = {len(raw_window)}")

#             # --- Append pulses with dynamic PRI ---
#             for row in raw_window:
#                 toa = row.get("toa", time.perf_counter())  # fallback if TOA missing

#                 if last_toa is None:
#                     pri = 0.0
#                 else:
#                     pri = toa - last_toa
#                 last_toa = toa

#                 pulse_buffer.append({
#                     "doa": row["doa"],
#                     "frequency": row["frequency"],
#                     "pulse_width": row["pulse_width"],
#                     "pri": pri
#                 })
#                 pulse_counter_since_cluster += 1
#                 total_pulses_received += 1

#             # Amortized clustering
#             if pulse_counter_since_cluster >= CLUSTER_BATCH_SIZE:
#                 process_buffer(pulse_buffer, MIN_SAMPLES)
#                 pulse_counter_since_cluster = 0

#             # Keep buffer bounded
#             if len(pulse_buffer) > WINDOW_SIZE:
#                 pulse_buffer = pulse_buffer[-WINDOW_SIZE:]

#     # -------- END-OF-STREAM FLUSH --------
#     if SOFT_MIN_SAMPLES <= len(pulse_buffer) < MIN_SAMPLES:
#         print("End-of-stream → running relaxed clustering")
#         process_buffer(pulse_buffer, SOFT_MIN_SAMPLES)
#     elif len(pulse_buffer) < SOFT_MIN_SAMPLES:
#         print("Remaining pulses discarded as noise")

#     end_time = time.perf_counter()

#     # -------- FINAL OUTPUT --------
#     print("\nFINAL RADAR EMITTER SUMMARY")
#     print(f"Total Pulses Received: {total_pulses_received}")
#     print(f"Total Emitters Detected: {len(emitters)}\n")

#     total_clustered_pulses = sum(e["pulseCount"] for e in emitters.values())
#     print(f"Total Pulses Clustered: {total_clustered_pulses}\n")

#     for eid, e in emitters.items():
#         print(
#             f"Emitter {eid} | "
#             f"DOA {e['meanDOA']:.2f}° "
#             f"[{e['minDOA']:.2f}-{e['maxDOA']:.2f}] | "
#             f"Freq {e['meanFreq']:.2f} MHz "
#             f"[{e['minFreq']:.2f}-{e['maxFreq']:.2f}] | "
#             f"Pulses {e['pulseCount']}"
#         )

#     print("\nTotal Processing Time:")
#     print(f"{end_time - start_time:.3f} seconds")



import socket
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time


# ---------------- VISUALIZATION ----------------
def visualize_radar_scene(emitters, ships):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    R_MAX = 300
    ax.set_xlim(-R_MAX, R_MAX)
    ax.set_ylim(-R_MAX, R_MAX)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # ---------------- RADAR ----------------
    ax.scatter(0, 0, c="lime", s=200, marker="^")
    ax.text(0, -15, "RADAR", color="lime", ha="center")

    # Range rings
    for r in [75, 150, 225, 300]:
        ax.add_artist(plt.Circle((0, 0), r, color="green", fill=False, alpha=0.15))

    # Radar sweep
    sweep_line, = ax.plot([], [], color="lime", lw=2, alpha=0.8)

    # ---------------- EMITTERS ----------------
    emitter_positions = {}
    emitter_blink = {}

    for eid, e in emitters.items():
        angle = np.deg2rad(e["meanDOA"])
        r = 80 + (e["meanFreq"] % 180)

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        emitter_positions[eid] = [x, y]
        emitter_blink[eid] = np.random.rand() * 2 * np.pi

    emitter_scatter = ax.scatter([], [], c="red", s=40)

    # ---------------- SHIPS ----------------
    ship_circles = []

    for sid, ship in enumerate(ships):
        xs = [emitter_positions[eid][0] for eid in ship["emitters"]]
        ys = [emitter_positions[eid][1] for eid in ship["emitters"]]

        cx, cy = np.mean(xs), np.mean(ys)
        radius = max(
            np.sqrt((np.array(xs) - cx) ** 2 + (np.array(ys) - cy) ** 2)
        ) + 12

        circle = plt.Circle((cx, cy), radius, color="red", fill=False, lw=2)
        ax.add_patch(circle)
        ship_circles.append(circle)

        ax.text(cx, cy, f"SHIP {sid}", color="red",
                fontsize=11, fontweight="bold", ha="center")

    # ---------------- ANIMATION ----------------
    theta = 0.0

    def update(frame):
        nonlocal theta
        theta += 0.03
        if theta > 2 * np.pi:
            theta = 0

        # Radar sweep
        x = R_MAX * np.cos(theta)
        y = R_MAX * np.sin(theta)
        sweep_line.set_data([0, x], [0, y])

        # Emitters blinking
        xs, ys, sizes = [], [], []

        for eid, pos in emitter_positions.items():
            emitter_blink[eid] += 0.15
            intensity = 1 + 0.7 * np.sin(emitter_blink[eid])

            xs.append(pos[0])
            ys.append(pos[1])
            sizes.append(50 * intensity)

        emitter_scatter.set_offsets(np.c_[xs, ys])
        emitter_scatter.set_sizes(sizes)

        # Ship slow movement (sea drift)
        for circle in ship_circles:
            cx, cy = circle.center
            circle.center = (
                cx + np.random.uniform(-0.15, 0.15),
                cy + np.random.uniform(-0.15, 0.15)
            )

        return sweep_line, emitter_scatter

    ani = FuncAnimation(
        fig,
        update,
        frames=600,
        interval=40,
        blit=False
    )

    plt.show()


# ---------------- CONFIG ----------------
HOST = "127.0.0.1"
PORT = 65432

WINDOW_SIZE = 5000
EPS = 0.8
MIN_SAMPLES = 20

DOA_GATE = 5.0
FREQ_GATE = 20.0

# Ship-level constraints
SHIP_DOA_GATE = 5.0
SHIP_FREQ_SPAN = 200.0
# --------------------------------------

# ---------------- STORAGE ----------------
pulse_buffer = deque(maxlen=WINDOW_SIZE)
emitters = {}
emitter_id_counter = 0
scaler = StandardScaler()

last_toa = None
# ----------------------------------------

# -------- EMITTER FUNCTIONS --------
def associate_emitter(stats):
    for eid, e in emitters.items():
        if abs(stats["meanDOA"] - e["meanDOA"]) < DOA_GATE and \
           abs(stats["meanFreq"] - e["meanFreq"]) < FREQ_GATE:
            return eid
    return None

def update_emitter(eid, stats):
    e = emitters[eid]
    e["meanDOA"]  = 0.7 * e["meanDOA"]  + 0.3 * stats["meanDOA"]
    e["meanFreq"] = 0.7 * e["meanFreq"] + 0.3 * stats["meanFreq"]
    e["minDOA"]   = min(e["minDOA"], stats["minDOA"])
    e["maxDOA"]   = max(e["maxDOA"], stats["maxDOA"])
    e["minFreq"]  = min(e["minFreq"], stats["minFreq"])
    e["maxFreq"]  = max(e["maxFreq"], stats["maxFreq"])
    e["pulseCount"] += stats["pulseCount"]

def create_emitter(stats):
    global emitter_id_counter
    emitters[emitter_id_counter] = {
        "meanDOA": stats["meanDOA"],
        "meanFreq": stats["meanFreq"],
        "minDOA": stats["minDOA"],
        "maxDOA": stats["maxDOA"],
        "minFreq": stats["minFreq"],
        "maxFreq": stats["maxFreq"],
        "pulseCount": stats["pulseCount"]
    }
    emitter_id_counter += 1

# -------- SHIP GROUPING --------
def group_emitters_into_ships(emitters):
    ships = []

    for eid, e in emitters.items():
        assigned = False

        for ship in ships:
            doa_close = abs(e["meanDOA"] - ship["meanDOA"]) < SHIP_DOA_GATE

            combined_min_freq = min(ship["minFreq"], e["minFreq"])
            combined_max_freq = max(ship["maxFreq"], e["maxFreq"])
            freq_span_ok = (combined_max_freq - combined_min_freq) < SHIP_FREQ_SPAN

            if doa_close and freq_span_ok:
                ship["meanDOA"] = 0.7 * ship["meanDOA"] + 0.3 * e["meanDOA"]
                ship["minFreq"] = combined_min_freq
                ship["maxFreq"] = combined_max_freq
                ship["emitters"].append(eid)
                assigned = True
                break

        if not assigned:
            ships.append({
                "meanDOA": e["meanDOA"],
                "minFreq": e["minFreq"],
                "maxFreq": e["maxFreq"],
                "emitters": [eid]
            })

    return ships

# -------- PROCESS WINDOW --------
def process_window(buffer):
    if len(buffer) < MIN_SAMPLES:
        return

    df = pd.DataFrame(buffer)

    X = scaler.fit_transform(
        df[["doa", "frequency", "pulse_width", "pri"]]
    )

    labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X)
    df["cluster"] = labels

    for cid in set(labels):
        if cid == -1:
            continue

        c = df[df["cluster"] == cid]

        stats = {
            "meanDOA": c["doa"].mean(),
            "meanFreq": c["frequency"].mean(),
            "minDOA": c["doa"].min(),
            "maxDOA": c["doa"].max(),
            "minFreq": c["frequency"].min(),
            "maxFreq": c["frequency"].max(),
            "pulseCount": len(c)
        }

        eid = associate_emitter(stats)
        if eid is None:
            create_emitter(stats)
        else:
            update_emitter(eid, stats)

# ---------------- MAIN CLIENT ----------------
if __name__ == "__main__":
    start_time = time.perf_counter()
    window_count = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        print("Connected to server")

        while True:
            length_bytes = client.recv(8)
            if not length_bytes:
                break

            data_length = int.from_bytes(length_bytes, "big")
            payload = b""

            while len(payload) < data_length:
                payload += client.recv(data_length - len(payload))

            raw_window = pickle.loads(payload)
            window_count += 1
            print(f"t{window_count} → pulses arrived = {len(raw_window)}")

            for row in raw_window:
                toa = row["toa"]

                if last_toa is None:
                    pri = 0.0
                else:
                    pri = toa - last_toa

                last_toa = toa

                pulse_buffer.append({
                    "doa": row["doa"],
                    "frequency": row["frequency"],
                    "pulse_width": row["pulse_width"],
                    "pri": pri
                })

            process_window(pulse_buffer)

    end_time = time.perf_counter()

    # ---------------- OUTPUT ----------------
    print("\nFINAL RADAR EMITTER SUMMARY")
    print(f"Total Emitters Detected: {len(emitters)}\n")

    for eid, e in emitters.items():
        print(
            f"Emitter {eid} | "
            f"DOA {e['meanDOA']:.2f}° "
            f"[{e['minDOA']:.2f}-{e['maxDOA']:.2f}] | "
            f"Freq {e['meanFreq']:.2f} MHz "
            f"[{e['minFreq']:.2f}-{e['maxFreq']:.2f}] | "
            f"Pulses {e['pulseCount']}"
        )

    ships = group_emitters_into_ships(emitters)

    print("\n--- SHIP LEVEL SUMMARY ---")
    print(f"Total number of emitters formed = {len(emitters)}")
    print(f"Possible number of ships = {len(ships)}")

    for i, ship in enumerate(ships):
        print(
            f"Ship {i} | "
            f"DOA ~ {ship['meanDOA']:.2f}° | "
            f"Freq Span [{ship['minFreq']:.2f}-{ship['maxFreq']:.2f}] MHz | "
            f"Emitters {ship['emitters']}"
        )

    print("\nTotal Processing Time:")
    print(f"{end_time - start_time:.3f} seconds")
    visualize_radar_scene(emitters, ships)




