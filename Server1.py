# radar_server.py
import socket
import pandas as pd
import pickle
import time

HOST = "127.0.0.1"
PORT = 65432
WINDOW_SIZE = 5000

if __name__ == "__main__":
    df = pd.read_csv("DF51.csv")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Server listening on {HOST}:{PORT}")

        conn, addr = server.accept()
        with conn:
            print("Client connected:", addr)

            for t_index, i in enumerate(range(0, len(df), WINDOW_SIZE), start=1):
                window_df = df.iloc[i:i + WINDOW_SIZE]
                payload = pickle.dumps(window_df.to_dict("records"))

                # send length + payload
                conn.sendall(len(payload).to_bytes(8, "big") + payload)
                print(f"t{t_index} → sent {len(window_df)} pulses")

                time.sleep(0.01)  # simulate real-time

            print("All data sent. Closing connection.")
