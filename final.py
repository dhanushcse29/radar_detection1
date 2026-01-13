import pandas as pd
import time
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


WINDOW_SIZE = 2000
EPS = 0.8
MIN_SAMPLES = 20
DOA_GATE = 5.0
FREQ_GATE = 20.0


# STORAGE
pulse_buffer = deque(maxlen=WINDOW_SIZE)
emitters = {}
emitter_id_counter = 0

scaler = StandardScaler()


# ASSOCIATE CLUSTER TO EMITTER
def associate_emitter(stats):
    for eid, e in emitters.items():
        if abs(stats["meanDOA"] - e["meanDOA"]) < DOA_GATE and \
           abs(stats["meanFreq"] - e["meanFreq"]) < FREQ_GATE:
            return eid
    return None


# UPDATE EMITTER
def update_emitter(eid, stats):
    e = emitters[eid]
    e["meanDOA"] = 0.7 * e["meanDOA"] + 0.3 * stats["meanDOA"]
    e["meanFreq"] = 0.7 * e["meanFreq"] + 0.3 * stats["meanFreq"]
    e["minDOA"] = min(e["minDOA"], stats["minDOA"])
    e["maxDOA"] = max(e["maxDOA"], stats["maxDOA"])
    e["minFreq"] = min(e["minFreq"], stats["minFreq"])
    e["maxFreq"] = max(e["maxFreq"], stats["maxFreq"])
    e["pulseCount"] += stats["pulseCount"]


# CREATE EMITTER
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


# PROCESS WINDOW
def process_window(buffer_list):
    if len(buffer_list) < MIN_SAMPLES:
        return

    df = pd.DataFrame(buffer_list)

    X = scaler.fit_transform(
        df[['doa', 'frequency', 'pulse_width', 'pri']]
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


if __name__ == "__main__":

    start_time = time.perf_counter()   

    df = pd.read_csv("Triple_NoCombined.csv")

    for i in range(0, len(df), WINDOW_SIZE):
        window = df.iloc[i:i + WINDOW_SIZE]
        process_window(window.to_dict("records"))

    end_time = time.perf_counter()     


    print("\n FINAL RADAR EMITTER SUMMARY")
    print(f" Total Emitters Detected: {len(emitters)}\n")

    for eid, e in emitters.items():
        print(
            f"Emitter {eid} | "
            f"DOA {e['meanDOA']:.2f}° "
            f"[{e['minDOA']:.2f}-{e['maxDOA']:.2f}] | "
            f"Freq {e['meanFreq']:.2f} MHz "
            f"[{e['minFreq']:.2f}-{e['maxFreq']:.2f}] | "
            f"Pulses {e['pulseCount']}"
        )

    print("\n Time Taken")
    print(f"Total Processing Time: {end_time - start_time:.3f} seconds")
