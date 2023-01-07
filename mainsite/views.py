from django.shortcuts import render


import os
import json
import pandas as pd
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
import json
import torch
from sentence_transformers import util

def cosine_sim(em1, em2):
    return util.cosine_sim(em1, em2)

embedding = TransformerDocumentEmbeddings('xlm-roberta-base')
embeddings = []

with open('mydata.json','r+') as f:
    embeddings = json.load(f)

for k in embeddings:
    k['embedding'] = torch.Tensor(k['embedding'])


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



# Create your views here.
def index(request):
    return render(request, 'mainsite/index.html')


def results(request):
    
    input_text = ""

    if request.method == "POST":
        input_text = request.POST['inpkeywords']
    
    if input_text is not None:
        input_text = input_text.replace(",","")

        inp = Sentence(input_text)
        embedding.embed(inp)

        inp_embedding = inp.get_embedding()

        sim = []

        for i in range (0, 100):
            sim.append(util.cos_sim(inp_embedding, embeddings[i]['embedding'])[0])


        result_list = N_max_elements(sim, 10)

        res_ids = []
        res_contents = []
        res_file_names = []
        for res in result_list:

            file_name = "documents/" + str(res+1) + ".txt"

            if(os.path.exists(file_name)):
                with open(file_name, "r", encoding="utf-8") as f:
                    contents = f.read()
                    res_contents.append(contents)
                    res_ids.append(res+1)
                    res_file_names.append(file_name)
            
        result_df = pd.DataFrame({'id': res_ids, 'content': res_contents, 'file_name': res_file_names})

        with open('res_data.json', 'w') as f:
            f.write(result_df.to_json(orient='records'))

        return render(request, 'mainsite/result.html', {'data':json.load(open('res_data.json'))})
    else:
        print("Text is empty")
        return None

        


