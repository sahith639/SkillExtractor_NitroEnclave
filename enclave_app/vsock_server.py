import socket
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


PORT = 5005
CID_ANY = socket.VMADDR_CID_ANY

def compute_similarity(input_syllabi_df, input_skills_df):
    skills_np = input_skills_df.iloc[:, 1:].to_numpy()
    input_syllabi_f_df = input_syllabi_df.drop(['syllabus_text'], axis=1)

    unique_ids = list(input_syllabi_f_df['syllabus_id'].unique())
    n_skills = skills_np.shape[0]

    final_max_similarities = np.zeros((len(unique_ids), n_skills), dtype='float16')
    ids = np.zeros([len(unique_ids)], dtype='int64')
    sentence_ids = np.zeros((len(unique_ids), n_skills), dtype='uint16')

    for idx, uid in enumerate(unique_ids):
        subset = input_syllabi_f_df[input_syllabi_f_df['syllabus_id'] == uid].reset_index(drop=True)
        vectors = subset.iloc[:, 2:].to_numpy()
        sims = cosine_similarity(vectors, skills_np)
        final_max_similarities[idx] = np.max(sims, axis=0)
        argmax_indices = np.argmax(sims, axis=0)
        sentence_ids[idx] = subset.loc[argmax_indices, 'sent_id'].to_list()
        ids[idx] = uid

    skill_cols = [str(i) for i in range(n_skills)]

    scores_df = pd.DataFrame(final_max_similarities, columns=skill_cols)
    scores_df.insert(0, 'syllabus_id', ids)

    sentence_ids_df = pd.DataFrame(sentence_ids, columns=skill_cols)
    sentence_ids_df.insert(0, 'syllabus_id', ids)

    # 🔍 Print top 10 skills per syllabus
    print("\n🔝 Top 10 Skills per Syllabus Based on Sentiment Score:")
    for idx, row in scores_df.iterrows():
        sid = row['syllabus_id']
        top_skills = row[1:].astype(float).nlargest(10)
        print(f"\n📘 Syllabus ID: {sid}")
        for i, (skill_idx, score) in enumerate(top_skills.items(), 1):
            print(f"   {i}. Skill {skill_idx} → Score: {score:.4f}")

    return {
        "scores": scores_df.to_json(orient='split'),
        "sentence_ids": sentence_ids_df.to_json(orient='split')
    }

def start_vsock_server():
    with socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM) as server:
        server.bind((CID_ANY, PORT))
        server.listen(1)
        print("Enclave: Listening on vsock port", PORT)

        conn, _ = server.accept()
        with conn:
            print("Enclave: Client connected")

            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"<END>" in data:
                    break

            data = data.replace(b"<END>", b"")
            payload = json.loads(data.decode())

            sent_df = pd.read_json(payload["sentences"])
            skills_df = pd.read_json(payload["skills"])

            result = compute_similarity(sent_df, skills_df)

            conn.sendall((json.dumps(result) + "<END>").encode())
            print("Enclave: Sent results back")

if __name__ == "__main__":
    start_vsock_server()
