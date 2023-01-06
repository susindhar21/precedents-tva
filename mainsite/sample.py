import os
import json
import pandas as pd

from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence

from sentence_transformers import util

input_text = "துறைசார் தொழிலாளர் முறை"

input_text = input_text.replace(",","")

embedding = TransformerDocumentEmbeddings('xlm-roberta-base')

embeddings = []

data = []

for i in range (1,102):
    file_name = "E:/Susindhar/Projects/Precedents_Research/documents/" + str(i) + ".txt"
    if(os.path.exists(file_name)):
        with open(file_name, "r", encoding="utf-8") as f:
            contents = f.read()
            data.append(contents)


for i in range (0, 101):
    
    sentence = Sentence(data[i])
    embedding.embed(sentence)
    
    embeddings.append(sentence.get_embedding())


def N_max_elements(list, N):
            result_list = []
        
            for i in range(0, N): 
                maximum = 0
                maxpos = 0
                for j in range(len(list)):     
                    if list[j] > maximum:
                        maximum = list[j]
                        maxpos = j
                        
                list.remove(maximum)
                result_list.append(maxpos)
                
            return result_list


inp = Sentence(input_text)
embedding.embed(inp)

inp_embedding = inp.get_embedding()

sim = []

for i in range (0, 100):
    sim.append(util.cos_sim(inp_embedding, embeddings[i])[0])


result_list = N_max_elements(sim, 10)

res_ids = []
res_contents = []
res_file_names = []
for res in result_list:

    file_name = "documents" + str(res+1) + ".txt"

    if(os.path.exists(file_name)):
        with open(file_name, "r", encoding="utf-8") as f:
            contents = f.read()
            res_contents.append(contents)
            res_ids.append(res+1)
            res_file_names.append(file_name)
    
result_df = pd.DataFrame({'id': res_ids, 'content': res_contents, 'file_name': res_file_names})

with open('res_data.json', 'w') as f:
    f.write(result_df.to_json(orient='records'))
