import time
import os
import pandas as pd
from utils import *
from settings import *

if __name__ == '__main__':
    for f_name in file_names_list:
        print("Start:", f_name)
        start = time.perf_counter()
        file_name = f_name.split('.')[0]  #Extract base name for intermediate file naming

        syll_df = pd.read_csv(os.path.join(inputs_path, f_name), encoding='utf-8', compression=comp_meth)
        syll_df['syllabus_id'] = syll_df['syllabus_id'].astype(int)

        # Step 2: Sentence segmentation
        syll_df = stanza_syl_2_lines_array(syll_df, intermediate_path, file_name, segmnt_intermediate_save)

        # Step 3: Outcome cleaning
        if with_bow_cleaning:
            syll_df = bag_of_word_tagger(syll_df, intermediate_path, file_name, bow_intermediate_save)

        # Step 4: Skill embedding
        n_skills = 676 if skill_type == 'dwa' else 18429
        onet_df = embedd_skill_type(onet_path, skill_type, bert_model)

        # Step 5: Sentence embedding
        syll_df = sentences_embedding(input_df=syll_df,
                                  bert_model=bert_model)

        # Step 6: Save for enclave
        syll_df.to_csv(f"../shared_data/sentences_embedded.csv", index=False)
        onet_df.to_csv(f"../shared_data/skills_embedded.csv", index=False)

        end = time.perf_counter()
        print(f"Done: {f_name} | Time taken: {end - start:.2f} sec")

