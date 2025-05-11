import socket
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import re
import io

PORT = 5005
HOST = "127.0.0.1"

# ✅ Local helper: split text into chunks (unused, but here for completeness)
def split_into_sentences(text: str):
    chunks = re.split(r'[.\n•;•\-•–,]', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# ✅ Local helper: literal match boost
def literal_skill_mentions(text_block: str, skills_list):
    mentioned = []
    lower_text = str(text_block).lower()
    for skill in skills_list:
        if skill.lower() in lower_text:
            mentioned.append(skill)
    return list(set(mentioned))


def compute_similarity(input_syllabi_df, input_skills_df, threshold=0.0, top_k=10, alpha=0.9, beta=0.1):
    skills = input_skills_df.iloc[:, 0].tolist()
    skills_np = input_skills_df.iloc[:, 1:].to_numpy().astype(np.float32)

    # 🛠 Fix column drop issue (don’t drop if 'syllabus_text' not there)
    input_syllabi_f_df = input_syllabi_df.drop(columns=['syllabus_text'], errors='ignore')

    # 🧠 Determine column that holds actual sentence strings
    if 'sentence' in input_syllabi_df.columns:
        text_column = 'sentence'
    elif 'syllabus_text' in input_syllabi_df.columns:
        text_column = 'syllabus_text'
    else:
        raise KeyError("No sentence text column found in input.")

    unique_ids = list(input_syllabi_f_df['syllabus_id'].unique())
    n_skills = skills_np.shape[0]

    final_scores = np.zeros((len(unique_ids), n_skills), dtype='float32')
    sentence_ids = np.zeros((len(unique_ids), n_skills), dtype='uint16')
    ids = np.zeros([len(unique_ids)], dtype='int64')

    skills_np_norm = normalize(skills_np, axis=1)

    for idx, uid in enumerate(unique_ids):
        subset = input_syllabi_f_df[input_syllabi_f_df['syllabus_id'] == uid].reset_index(drop=True)
        vectors = subset.iloc[:, 2:].to_numpy().astype(np.float32)
        sent_texts = input_syllabi_df[input_syllabi_df['syllabus_id'] == uid][text_column].tolist()
        vectors_norm = normalize(vectors, axis=1)

        sims = np.dot(vectors_norm, skills_np_norm.T)

        # Boost literal matches
        boost_matrix = np.zeros_like(sims)
        for i, sentence_text in enumerate(sent_texts):
            matched = literal_skill_mentions(sentence_text, skills)
            for sk in matched:
                if sk in skills:
                    skill_idx = skills.index(sk)
                    boost_matrix[i, skill_idx] = 1.0

        hybrid_sims = alpha * sims + beta * boost_matrix

        final_scores[idx] = np.max(hybrid_sims, axis=0)
        argmax_indices = np.argmax(hybrid_sims, axis=0)
        sentence_ids[idx] = subset.loc[argmax_indices, 'sent_id'].to_list()
        ids[idx] = uid

    skill_cols = [str(i) for i in range(n_skills)]

    scores_df = pd.DataFrame(final_scores, columns=skill_cols)
    scores_df.insert(0, 'syllabus_id', ids)

    sentence_ids_df = pd.DataFrame(sentence_ids, columns=skill_cols)
    sentence_ids_df.insert(0, 'syllabus_id', ids)

    print("\n🔝 Top Matching Skills per JD Based on Hybrid Scoring:")
    for idx, row in scores_df.iterrows():
        sid = row['syllabus_id']
        top_skills = row[1:].astype(float)
        top_skills = top_skills[top_skills >= threshold].nlargest(top_k)
        print(f"\n📄 JD ID: {sid}")
        for i, (skill_idx, score) in enumerate(top_skills.items(), 1):
            print(f"   {i}. Skill {skill_idx} → Score: {score:.4f}")
        if top_skills.empty:
            print("   ⚠️ No skills passed threshold.")
    top_skills_dict = {}
    for idx, row in scores_df.iterrows():
        sid = row['syllabus_id']
        top_indices = row[1:].astype(float).nlargest(10).index.tolist()  # Top 10 skill indices (column names)
        top_skills_dict[int(sid)] = [int(idx) for idx in top_indices]  # Convert string indices to int

    return {
        "scores": scores_df.to_json(orient='split'),
        "sentence_ids": sentence_ids_df.to_json(orient='split'),
        "top_skills": top_skills_dict  # 🆕 added
    }

def start_local_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Local Server: Listening on {HOST}:{PORT}")

        conn, _ = server.accept()
        with conn:
            print("Local Server: Client connected")

            data = b""
            total_received = 0
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                total_received += len(chunk)
                print(f"Server: Received {total_received} bytes...")
                data += chunk
                if b"<END>" in data:
                    break

            data = data.replace(b"<END>", b"")
            payload = json.loads(data.decode())

            import io
            sent_df = pd.read_json(io.StringIO(payload["sentences"]))
            skills_df = pd.read_json(io.StringIO(payload["skills"]))

            print("📥 Dataframes received. Running similarity computation...")
            result = compute_similarity(sent_df, skills_df)

            print("📤 Sending results back to client...")
            conn.sendall(json.dumps(result).encode())
            print("✅ Local Server: Done.")

if __name__ == "__main__":
    start_local_server()
