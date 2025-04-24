import socket
import json
import pandas as pd

PORT = 5005
ENCLAVE_CID = 16  # Default CID assigned to Nitro Enclave

def send_data_to_enclave():
    sentences_df = pd.read_csv("shared_data/sentences_embedded.csv")
    skills_df = pd.read_csv("shared_data/skills_embedded.csv")

    payload = {
        "sentences": sentences_df.to_json(),
        "skills": skills_df.to_json()
    }

    with socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM) as client:
        client.connect((ENCLAVE_CID, PORT))
        print("Client: Connected to enclave")

        payload_bytes = (json.dumps(payload) + "<END>").encode()
        total = len(payload_bytes)
        sent = 0
        while sent < total:
            chunk = payload_bytes[sent:sent+4096]
            client.send(chunk)
            sent += len(chunk)
            print(f"Client: Sent {sent}/{total} bytes")

        response = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            response += chunk
            if b"<END>" in response:
                break

        response = response.replace(b"<END>", b"")
        result = json.loads(response.decode())

        scores_df = pd.read_json(result["scores"], orient="split")
        sentence_ids_df = pd.read_json(result["sentence_ids"], orient="split")

        scores_df.to_csv("shared_data/final_scores.csv", index=False)
        sentence_ids_df.to_csv("shared_data/final_sentence_ids.csv", index=False)
        print("Client: Results saved to CSVs")

if __name__ == "__main__":
    send_data_to_enclave()
