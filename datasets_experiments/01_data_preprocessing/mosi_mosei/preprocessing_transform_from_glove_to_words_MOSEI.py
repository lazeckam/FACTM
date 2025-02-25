# See preprocessing_transform_from_glove_to_words.ipynb for details

import pickle
from tqdm import tqdm
import numpy as np
import sys

N_seq = [0, 7620, 15240, 22860]
i = int(float(sys.argv[1]))

print('computing from ', i)

file_path = './MultiBench/datasets_download'
words_dict = []
glove_dict = []

count = 0
words_dic = []
glove_dic = []

file_name = './glove_representation/glove.840B.300d.txt'

txt_file = open(file_name)

for line in tqdm(txt_file):
    count += 1
    tmp0 = line.split()
    start_index = len(tmp0) - 300
    words_dict.append(''.join(tmp0[0:start_index]))
    glove_dict.append(np.array(tmp0[start_index:(start_index+5)]).astype('float'))

txt_file.close()
glove_dict = np.array(glove_dict)
count


with open(file_path+'/'+str('MOSEI/MOSEI_transformed')+'.pkl', "rb") as input_file:
    dataset = pickle.load(input_file)
text_modality_glove = dataset['M2']
text_modality = []
N = len(text_modality_glove)
new_dict = np.array([[1000, 1000, 1000, 1000, 1000]])
new_words = ['word_wrong']
where_new_words = [""]

for i in tqdm(range(N_seq[i], N_seq[i+1])):

    words_tmp = text_modality_glove[i][:,0:5]

    text_tmp=""

    for j in range(words_tmp.shape[0]):
        
        word_tmp = words_tmp[j,:]

        distance_word_dict = np.sum((glove_dict - word_tmp)**2, axis=1)
        indeces_tmp = np.where(distance_word_dict == np.min(distance_word_dict))[0]

        # if too many fits
        if np.sum(distance_word_dict <= np.min(distance_word_dict) + 1e-10) > 1:
            print("Too many words fit glove rep.")

        if indeces_tmp.shape[0] > 1:
            print(str(i)+" "+str(j) + " more than one fits")

        index_tmp = indeces_tmp.astype('int')[0]

        if np.sum((glove_dict[index_tmp,:] - word_tmp)**2) > 1e-5:
            print(str(i)+" "+str(j)+" not found in a dictionary " + str(word_tmp))

            distance_new_word_dict = np.sum((new_dict - word_tmp)**2, axis=1)
            indeces_tmp = np.where(distance_new_word_dict == np.min(distance_new_word_dict))[0]

            index_tmp = indeces_tmp.astype('int')[0]

            if np.sum((new_dict[index_tmp,:] - word_tmp)**2) > 1e-5:
                print("-- Adding new word")
                new_dict = np.vstack((new_dict, word_tmp))
                new_words.append("WORD_"+str(len(new_dict) - 2))
                text_word_tmp = "WORD_"+str(len(new_dict) - 2)
                where_new_words.append([[i, j]])
            else:
                print("--using already added words")
                text_word_tmp = new_words[index_tmp]
        else:
            text_word_tmp = words_dict[index_tmp]
        
        text_tmp = text_tmp+" "+text_word_tmp

    text_modality.append(text_tmp)

all_info = [text_modality, new_dict, new_words, where_new_words]

with open('data_transformed/MOSEI/text_modality_v'+str(i)+'.pkl', 'wb') as f:  
    pickle.dump(all_info, f, protocol=5)