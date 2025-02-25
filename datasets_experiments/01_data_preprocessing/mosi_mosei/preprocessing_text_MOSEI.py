import pickle
from tqdm import tqdm
import numpy as np
import spacy
nlp = spacy.load("en_core_web_trf")
import sys

N_seq = [7620, 15240, 22860]
i = int(float(sys.argv[1]))

print('computing from ', i)

with open('data_transformed/MOSEI/text_modality_v'+str((N_seq[i]-1))+'.pkl', "rb") as input_file:
    text_modality = pickle.load(input_file)
text_modality = text_modality[0]

N = len(text_modality)
stop_words_part_of_speech = ['DET', 'PRON', 'CONJ', 'CCONJ', 'ADP']
text_modality_token = []
for n in tqdm(range(N)):
    text_modality_tmp = text_modality[n]
    text_modality_tmp = (' ').join([w for w in text_modality_tmp.split() if not 'youngentrepreneur.com' in w])
    nlp_n = nlp(text_modality_tmp[n])
    nlp_n = [nlp_n_w for nlp_n_w in nlp_n if not nlp_n_w.pos_ in stop_words_part_of_speech]
    text_modality_token.append(' '.join([nlp_n_w.lemma_ for nlp_n_w in nlp_n]))

with open('data_transformed/MOSEI/text_modality_token_v'+str(i)+'.pkl', 'wb') as f:  
    pickle.dump(text_modality_token, f, protocol=5)