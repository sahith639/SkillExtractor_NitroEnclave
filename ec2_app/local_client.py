import socket
import json
import pandas as pd

HOST = "127.0.0.1"
PORT = 5005

def send_data_to_local_server():
    # Load embedded CSVs
    sentences_df = pd.read_csv("shared_data/sentences_embedded.csv")
    skills_df = pd.read_csv("shared_data/skills_embedded.csv")

    payload = {
        "sentences": sentences_df.to_json(),
        "skills": skills_df.to_json()
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        print("Client: Connected to local server")

        payload_bytes = (json.dumps(payload) + "<END>").encode()
        total = len(payload_bytes)
        print(f"Client: Sending {total} bytes...")

        sent = 0
        chunk_size = 4096
        while sent < total:
            chunk = payload_bytes[sent:sent+chunk_size]
            client.send(chunk)
            sent += len(chunk)
            print(f"Client: Sent {sent}/{total} bytes")


        response = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            response += chunk

        result = json.loads(response.decode())
        scores_df = pd.read_json(result["scores"], orient="split")
        sentence_ids_df = pd.read_json(result["sentence_ids"], orient="split")

        scores_df.to_csv("shared_data/final_scores.csv", index=False)
        sentence_ids_df.to_csv("shared_data/final_sentence_ids.csv", index=False)
        print("Client: Results saved to CSVs")

if __name__ == "__main__":
    send_data_to_local_server()
