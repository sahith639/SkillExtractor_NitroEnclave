import stanza
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

import os
import sys


# Step 2: Sentence Segmentation
def stanza_syl_2_lines_array(input_df, intermediate_path, file_name, segmnt_intermediate_save=True):

    """" Function for parsing the syllabi and save one row for each line.
         - Inputs:
         -- input_df: The dataframe containing the course syllabi text and their ids.
         -- intermediate_path: path to save the intermediate results of the sentence segmentation step.
         -- file_name: The current CSV file name to be used for saving the intermediate result.
         -- segmnt_intermediate_save: If ture, saves the intermediate segmented sentences.
         - Outputs: A dataframe containing the lines of syllabi and save the results in gzip format for future refrence.
         """
    input_nlp = stanza.Pipeline(lang='en', processors='tokenize', 
                            tokenize_no_ssplit=False, use_gpu=True, 
                            verbose=False)

    # Basic pre-processing
    ## Removing non-ascii characters
    input_df['syllabus_text'] = input_df.apply(lambda row: row['syllabus_text'].encode("ascii", "ignore"), axis=1)
    input_df['syllabus_text'] = input_df.apply(lambda row: row['syllabus_text'].decode("ascii", "ignore"), axis=1)
    ## Replacing white space characters with space
    input_df['syllabus_text'] = input_df.apply(
        lambda row: row['syllabus_text'].replace('\\n', ' ').replace('\n', ' ').replace('\t', ' ').replace('\t',
                                                                                                           ' ').replace(
            '\r', ' ').replace('\x0b', ' ').replace('\x0c', ' '), axis=1)
    ## replace multiple white spaces with one white space
    input_df['syllabus_text'] = input_df.apply(lambda row: " ".join(row['syllabus_text'].split()), axis=1)

    lines_separated_dfs_list = []

    for idx in range(len(input_df)):
        this_osp_id = input_df.loc[idx, 'syllabus_id']
        this_syll = input_df.loc[idx, 'syllabus_text']
        this_doc = input_nlp(this_syll)
        all_sentences = [sentence.text for sentence in this_doc.sentences]
        temp_df = pd.DataFrame()
        temp_df['syllabus_text'] = all_sentences
        temp_df['syllabus_id'] = this_osp_id
        temp_df = temp_df[['syllabus_id', 'syllabus_text']]
        temp_df['sent_id'] = [i for i in range(len(temp_df))]
        lines_separated_dfs_list.append(temp_df)

    df_result = pd.concat(lines_separated_dfs_list, ignore_index=True)

    if segmnt_intermediate_save:
        os.makedirs(intermediate_path, exist_ok=True)
        df_result.to_csv(os.path.join(intermediate_path, 'sentSegIntermediate_' + file_name + '.gzip'),
                         compression='gzip', index=False)

    return df_result


# Step 3: General and Outcome sentences filtering
def bag_of_word_tagger(input_df, intermediate_path, file_name, bow_intermediate_save=True):
    """ Function for tagging OSP lines according to the list of Outcome General words/phrases.
        - Inputs:
        -- input_df: The segmented course syllabi df.
        -- intermediate_path: Path to save the intermediate results of the sentence tagging step.
        -- file_name: The current CSV file name to be used for saving the intermediate result.
        -- bow_intermediate_save: If ture, saves the intermediate sentences tagged as General/Outcome."""

    input_df['General'] = 0
    input_df['Outcome'] = 0

    general_terms_str_all = "Equal Opportunity|background check|drug screen|criminal record|identity verification|employer will consider|qualified applicants|disability|veteran status|diversity and inclusion|ADA|employment at will|affirmative action|benefits package|apply now|application process|EOE|401k|insurance coverage|dental|vision|medical|work authorization|must be authorized to work|schedule flexibility|hiring process|employment history|resume required|reference check|we are an equal opportunity employer|working conditions|compliance with policies|accommodation request|company values|background investigation|employment is contingent|must pass background check|safety regulations|physical demands|lifting up to|walking or standing|frequent movement|vaccination|COVID-19 guidelines|remote work policy|internal applicants only|sponsored role|unpaid internship|company description|startup culture|fast-paced|opportunity to grow|exciting environment|looking for passionate individuals|amazing team|inclusive workplace|company culture|compensation package|relocation assistance|flexible schedule|paid time off|PTO|disclaimer|legal authorization|signature required|mandatory training|employee handbook|company policy|standard operating procedure|HR policy|performance evaluation"

    input_df.loc[input_df['syllabus_text'].str.contains(general_terms_str_all, case=False), 'General'] = 1

    outcome_terms_str_all = "learn|demonstrate|develop|gain|opportunity to|experience with|experience in|professional experience|exposure to|assist|support|collaborate|participate|will be involved|train|training|growth opportunity|ability to|become proficient|hands-on|shadow|will have the chance to|we will teach|you will learn|on-the-job training|career development|knowledge of|understanding of|entry level|junior level|fast learner|motivated|adaptable|team player|problem solver|detail-oriented|communication skills|interpersonal skills|positive attitude|goal-oriented|multi-tasking|self-motivated|initiative-taking|critical thinking|passion for learning"

    input_df.loc[input_df['syllabus_text'].str.contains(outcome_terms_str_all, case=False), 'Outcome'] = 1

    if bow_intermediate_save:
        input_df.to_csv(os.path.join(intermediate_path, 'sentTaggedIntermediate_' + file_name + '.gzip'),
                        compression='gzip', index=False)

    input_df = input_df[(input_df['General'] == 0) & (input_df['Outcome'] == 1)].reset_index(drop=True)
    input_df = input_df.drop(['General', 'Outcome'], axis=1)

    return input_df


# Step 4: O*NET Skills Embedding
def embedd_skill_type(onet_path, skill_type, bert_model):
    """ Function to check if the skills (DWA/Task) embeddings (dimensions) exist or not. If not, create them.
    - Inputs:
    -- onet_path: O*NET data path.
    -- skill_type: Skill type to be used for mapping.
    -- bert_model: SBERT langauge model. """

    skills_with_embedding_path = os.path.join(onet_path, skill_type + '_' + bert_model + '.csv')
    skills_without_embedding_path = os.path.join(onet_path, skill_type + '.csv')

    if not os.path.isfile(skills_without_embedding_path):
        raise Exception("Please upload the ONET file in the folder.")

    if not os.path.isfile(skills_with_embedding_path):

        input_df = pd.read_csv(skills_without_embedding_path, encoding='utf-8')

        model = SentenceTransformer(bert_model)

        if skill_type == 'dwa':
            column_name = 'DWA Title'
        elif skill_type == 'task':
            column_name = 'Task'

        embeddings = model.encode(input_df[column_name], show_progress_bar=False)

        embeddings_space_df = pd.DataFrame(embeddings)

        final_df = pd.merge(input_df, embeddings_space_df, left_index=True, right_index=True)

        final_df.to_csv(skills_with_embedding_path, index=False)

    else:
        final_df = pd.read_csv(skills_with_embedding_path)

    return final_df


# Step 5: Embedding the sentences
def sentences_embedding(input_df, bert_model):
    """ Function for extracting the embeddings of a given syllabi CSV file
        Inputs:
        - input_df: Input df.
        - bert_model_name: SBERT model to be used for embedding.

        Output:
        - A dataframe file containing the sentences and the dimensions. """

    model = SentenceTransformer(bert_model)
    # print("Model loaded ...")

    embeddings = model.encode(input_df['syllabus_text'], show_progress_bar=False)
    # print("Embedding done ...")

    embeddings_space_df = pd.DataFrame(embeddings)

    final_df = pd.merge(input_df, embeddings_space_df, left_index=True, right_index=True)

    return final_df


# Step 6: Skills similarities
def skills_similarities(input_syllabi_df, input_skills_df, outputs_path, input_file_name,
                        save_relevant_sent_ids, n_skills, skill_type, bert_model, smilarities_dtype='float16'):
    """ Function for computing the similarities between Skills (DWAs/Tasks)
        and sentences and return the MAX similarity.
        - Inputs:
        -- input_syllabi_df: The syllabi daframe with their language model embedding dimenssions.
        -- input_skills_df: O*NET skills (DWA/Task) daframe with their language model embedding dimenssions.
        -- outputs_path: Output path to save the results.
        -- input_file_name: File name to be used for saving the results.
        -- save_relevant_sent_ids: If true, saves the sentence ids with the maximum score for
            each DWA/Task in a seprace file. The columns of the resulting CSV file correspond to
            the DWA/TAsk ids (input files in 'onet_path').
        -- n_skills: Number of skills according to DWA or Task.
        -- skill_type: O*NET Skills to be used as the refrence 'dwa' or 'task'.
        -- bert_model:SBERT language model
        -- smilarities_dtype: Similarity scores datatype. 'float16', 'float32', or 'float64'.

        - Outputs:
        -- Gzip containing the Final Scores: The naming convention ->
            input_file_name + '_FinalScores_' + skill_type +'_'+ bert_model +'.gzip'
        -- Gzip containing the Sentence IDs of the maximum Final Scores
            (Optional, will be genrated if save_relevant_sent_ids=True): The naming convention ->
            input_file_name + '_FinalSentIds_' + skill_type +'_'+ bert_model +'.gzip'
        *Note: Both outputs will be saved at 'outputs_path'. """

    skills_np = input_skills_df.iloc[:, 1:].to_numpy()

    input_syllabi_f_df = input_syllabi_df.drop(['syllabus_text'], axis=1)

    unique_ids = list(input_syllabi_f_df['syllabus_id'].unique())

    final_max_similarities = np.zeros([len(unique_ids), skills_np.shape[0]], dtype=smilarities_dtype)

    ids = np.zeros([len(unique_ids)], dtype='int64')

    sentene_ids = np.zeros([len(unique_ids), len(skills_np)], dtype='uint16')

    for syll_idx in range(len(unique_ids)):
        uid = unique_ids[syll_idx]
        this_syll_df = input_syllabi_f_df[input_syllabi_f_df['syllabus_id'] == uid].reset_index(drop=True)
        this_syll_np = this_syll_df.iloc[:, 2:].to_numpy()
        this_syll_similarities = cosine_similarity(this_syll_np, skills_np)
        max_similarities_np = np.max(this_syll_similarities, axis=0)

        argmax_indices = np.argmax(this_syll_similarities, axis=0)
        sentence_indices = this_syll_df.loc[argmax_indices, 'sent_id'].to_list()

        final_max_similarities[syll_idx, :] = max_similarities_np
        ids[syll_idx] = uid
        sentene_ids[syll_idx, :] = sentence_indices

    skills_cols_names = [i for i in range(n_skills)]

    scores_df = pd.DataFrame(columns=['syllabus_id'] + skills_cols_names, index=range(sentene_ids.shape[0]))
    scores_df['syllabus_id'] = ids
    scores_df.loc[:, 0:] = final_max_similarities

    final_save_path_scores = os.path.join(outputs_path,
                                          input_file_name + '_FinalScores_' + skill_type + '_' + bert_model + '.gzip')
    print("Final Sent Scores: ")
    print(scores_df)

    scores_df.to_csv(final_save_path_scores, index=False, compression='gzip')
    

    if save_relevant_sent_ids:
        rele_sentIds_df = pd.DataFrame(columns=['syllabus_id'] + skills_cols_names, index=range(sentene_ids.shape[0]))
        rele_sentIds_df['syllabus_id'] = ids
        rele_sentIds_df.loc[:, 0:] = sentene_ids

        final_save_path_sentIds = os.path.join(outputs_path,
                                               input_file_name + '_FinalSentIds_' + skill_type + '_' + bert_model + '.gzip')
        print("Final Sent IDs: ")
        print(rele_sentIds_df.head())
        rele_sentIds_df.to_csv(final_save_path_sentIds, index=False, compression='gzip')