import socket
import json
import pandas as pd
from fpdf import FPDF

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

        result = json.loads(response.decode())
        # Read back scores and sentence matches
        import io
        scores_df = pd.read_json(io.StringIO(result["scores"]), orient="split")
        sentence_ids_df = pd.read_json(io.StringIO(result["sentence_ids"]), orient="split")

        dwa_df = pd.read_csv("ONET_Data/dwa.csv")
        if "DWA Title" not in dwa_df.columns:
            raise ValueError(" 'DWA Title' column not found in DWA.csv")

        #  Server-provided top 10 skill indices
        top_skills_map = result.get("top_skills", {})

        scores_df.to_csv("shared_data/final_scores.csv", index=False)
        sentence_ids_df.to_csv("shared_data/final_sentence_ids.csv", index=False)
        print("Client: Results saved to CSVs")
        # Generate PDF summarizing Syllabus and Top Skills
        generate_pdf_from_syllabus_and_skills("./inputs/Test_Data.csv", top_skills_map, dwa_df)
#  PDF generation function
def generate_pdf_from_syllabus_and_skills(test_data_path, top_skills_map, dwa_df):
    jd_df = pd.read_csv(test_data_path)

    class UnicodePDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 16)
            self.cell(0, 10, "Course Syllabus & Top Skills", ln=True, align="C")
            self.ln(5)

        def job_section(self, jd_id, jd_text, skills, sentences):
            self.set_font("Arial", "B", 13)
            self.set_text_color(30, 30, 30)
            self.cell(0, 10, f"Syllabus {jd_id + 1}", ln=True)

            self.set_font("Arial", "", 11)
            safe_text = jd_text.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 8, safe_text)

            self.set_font("Arial", "BI", 11)
            self.set_text_color(60, 60, 60)
            self.cell(0, 8, "Top 10 Skills Extracted by Our Skill Extraction Pipeline:", ln=True)
            self.set_font("Arial", "", 11)
            self.set_text_color(0, 0, 0)
            for skill,sentence in zip(skills,sentences):
                safe_skill = skill.encode('latin-1', 'replace').decode('latin-1')
                self.set_font("Arial", "B", 11)
                self.cell(0, 8, f"- {safe_skill}", ln=True)
                sent = sentence.encode('latin-1', 'replace').decode('latin-1')
                self.set_font("Arial", "", 11)
                self.multi_cell(0, 8, f"Sentence - {sent}")
            self.ln(4)

    def get_top_skills(indices):
        names = []
        for idx in indices:
            try:
                names.append(dwa_df.iloc[idx]["DWA Title"])
            except Exception:
                names.append(f"[Invalid Skill ID {idx}]")
        return names
    def get_relevant_sentences(indices, jd_id):
        sentences = [] 
        sent_id_df = pd.read_csv("shared_data/final_sentence_ids.csv")
        sent_embedded_df = pd.read_csv("shared_data/sentences_embedded.csv")
        for idx in indices:
            try:
                sent_id = sent_id_df.iloc[jd_id][idx+1]
                sentence = sent_embedded_df[(sent_embedded_df['syllabus_id'] == jd_id+1) & (sent_embedded_df['sent_id'] == sent_id)]['syllabus_text']
                if not sentence.empty:
                    sentences.append(sentence.values[0])
                    print(f"Sentence ID {sent_id} for JD ID {jd_id + 1}: {sentence.values[0]}")
                else:
                    sentences.append(f"[No Sentence Found for ID {idx}]")
                       
            except Exception:
                sentences.append(f"[Invalid Sentence ID {idx}]")
        return sentences

    pdf = UnicodePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for jd_id, row in jd_df.iterrows():
        jd_text = row["syllabus_text"]
        skill_indices = top_skills_map.get(str(jd_id+1), [])
        print(f"Processing Syllabus ID {jd_id + 1} with {len(skill_indices)} skills")
        skill_names = get_top_skills(skill_indices)
        sentences = get_relevant_sentences(skill_indices, jd_id)
        pdf.job_section(jd_id, jd_text, skill_names, sentences)

    output_path = "shared_data/CourseSyllabus_TopSkills.pdf"
    pdf.output(output_path)
    print(f"PDF successfully created at: {output_path}")

if __name__ == "__main__":
    send_data_to_enclave()
