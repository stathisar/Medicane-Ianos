from gr_nlp_toolkit import Pipeline
import pandas as pd
import numpy as np
import csv
nlp = Pipeline("ner")
data = pd.read_csv("/media/stathis/StathisUSB/final_classification_march_9/text_data/DataTextAllSources_March_15_Removed_duplicates.csv")
data['ner'] = ''
data['text'] = data['text'].fillna('')
data['translated'] = data['translated'].fillna('')

i = 0
v = list()
s = list()

for i in range(0, len(data)):
    text = data.loc[i, 'translated']
    
    if len(text) > 1100:
        midpoint = len(text) // 2
        text1 = text[:midpoint]
        text2 = text[midpoint:]
        text = text1
        
        doc = nlp(text)
        for token in doc.tokens:
            if token.ner in ("S-LOC", "B-LOC", "I-LOC", "S-GPE", "B-GPE", "I-GPE", "E-GPE", "E-LOC"):
                v.append(i)
                v.append(token.text)
                v.append(token.ner)
        
        text = text2
        for token in doc.tokens:
            if token.ner in ("S-LOC", "B-LOC", "I-LOC", "S-GPE", "B-GPE", "I-GPE", "E-GPE", "E-LOC"):
                v.append(i)
                v.append(token.text)
                v.append(token.ner)
        
        data.loc[i, 'ner'] = str(v)
        v = list()
    
    elif len(text) < 1101:
        doc = nlp(text)
        
        for token in doc.tokens:
            if token.ner in ("S-LOC", "B-LOC", "I-LOC", "S-GPE", "B-GPE", "I-GPE", "E-GPE", "E-LOC"):
                v.append(i)
                v.append(token.text)
                v.append(token.ner)
        
        data.loc[i, 'ner'] = str(v)
        s.append(v)
    v = list()


with open('/media/stathis/StathisUSB/final_classification_march_9/results/ler/ler.csv', 'w') as f:
	write = csv.writer(f)
	i = 0
	for i in range(0, len(s)):
		write.writerow(s[i])

data.to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/ler/data.csv')
data.sample(n=100).to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/ler/data_sample.csv')
