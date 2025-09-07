from gr_nlp_toolkit import Pipeline
import pandas as pd
import numpy as np
import csv
nlp = Pipeline("ner")
data = pd.read_csv("/home/stathis/Desktop/a little map/final_classification_march_9/text_data/DataTextAllSources_March_15_Removed_duplicates.csv")
data['ner'] = ''
data['text'] = data['text'].fillna('')
data['translated'] = data['translated'].fillna('')

i = 0
j = 0
#v = list()
#s = list()
d = {'ID': [], 'location': [], 'ner':[], 'i':[], 'text': [], 'translated': [], 'source_id': [], 'date_time': [], 'photo_video_id': [], 'photofilename': []}

data_location = pd.DataFrame(data = d, dtype = str)
data_location_text = pd.DataFrame(data = d, dtype = str)


for i in range(0, len(data)):
    try:
        text1 = nlp(data.loc[i, 'translated'])
        j = 0
        for j in range(0, len(text1.tokens)):
            if (text1.tokens[j].ner == 'S-GPE'):
                data_location_text.loc[1, 'ner'] = 'S-GPE'
                data_location_text.loc[1, 'location'] = str(text1.tokens[j].text)
                data_location_text.loc[1, 'ID'] = str(data.loc[i, 'ID'])
                data_location_text.loc[1, 'i'] = i
                data_location_text.loc[1, 'text'] = str(data.loc[i, 'text'])
                data_location_text.loc[1, 'translated'] = str(data.loc[i, 'translated'])
                data_location_text.loc[1, 'source_id'] = str(data.loc[i, 'source_id'])
                data_location_text.loc[1, 'date_time'] = str(data.loc[i, 'date_time'])
                data_location_text.loc[1, 'photo_video_id'] = str(data.loc[i, 'photo_video_id'])
                data_location_text.loc[1, 'photofilename'] = str(data.loc[i, 'photofilename'])
                data_location = pd.concat([data_location, data_location_text])
                data_location_text = pd.DataFrame(data = d, dtype = str)           
                print(i)                
            elif (text1.tokens[j].ner == 'S-LOC'):
                data_location_text.loc[1, 'ner'] = 'S-LOC'
                data_location_text.loc[1, 'location'] = str(text1.tokens[j].text)
                data_location_text.loc[1, 'ID'] = str(data.loc[i, 'ID'])
                data_location_text.loc[1, 'i'] = i
                data_location_text.loc[1, 'text'] = str(data.loc[i, 'text'])
                data_location_text.loc[1, 'translated'] = str(data.loc[i, 'translated'])
                data_location_text.loc[1, 'source_id'] = str(data.loc[i, 'source_id'])
                data_location_text.loc[1, 'date_time'] = str(data.loc[i, 'date_time'])
                data_location_text.loc[1, 'photo_video_id'] = str(data.loc[i, 'photo_video_id'])
                data_location_text.loc[1, 'photofilename'] = str(data.loc[i, 'photofilename'])
                data_location = pd.concat([data_location, data_location_text])
                data_location_text = pd.DataFrame(data = d, dtype = str)                           
                print(i)            
            elif (text1.tokens[j].ner == 'B-LOC'):
                location = list()
                ner = list()
                location.append(text1.tokens[j].text)
                ner.append(text1.tokens[j].ner)
                z = int(j + 1)
                while (text1.tokens[z].ner != 'E-LOC'):
                    location.append(text1.tokens[z].text)
                    ner.append(text1.tokens[z].ner)
                    z += 1
                location.append(text1.tokens[z].text)
                ner.append(text1.tokens[z].ner)
                data_location_text.loc[1, 'ner'] = ' '.join(ner)
                data_location_text.loc[1, 'location'] = ' '.join(location)
                ner = list()
                location = list()
                data_location_text.loc[1, 'ID'] = str(data.loc[i, 'ID'])
                data_location_text.loc[1, 'i'] = i
                data_location_text.loc[1, 'text'] = str(data.loc[i, 'text'])
                data_location_text.loc[1, 'translated'] = str(data.loc[i, 'translated'])
                data_location_text.loc[1, 'source_id'] = str(data.loc[i, 'source_id'])
                data_location_text.loc[1, 'date_time'] = str(data.loc[i, 'date_time'])
                data_location_text.loc[1, 'photo_video_id'] = str(data.loc[i, 'photo_video_id'])
                data_location_text.loc[1, 'photofilename'] = str(data.loc[i, 'photofilename'])                 
                data_location = pd.concat([data_location, data_location_text])
                data_location_text = pd.DataFrame(data = d, dtype = str)           
                print(i)
            elif (text1.tokens[j].ner == 'B-GPE'):
                location = list()
                ner = list()
                location.append(text1.tokens[j].text)
                ner.append(text1.tokens[j].ner)
                z = int(j + 1)
                while (text1.tokens[z].ner != 'E-GPE'):
                    location.append(text1.tokens[z].text)
                    ner.append(text1.tokens[z].ner)
                    z += 1
                    if (z - j > 15):
                        break
                location.append(text1.tokens[z].text)
                ner.append(text1.tokens[z].ner)
                data_location_text.loc[1, 'ner'] = ' '.join(ner)
                data_location_text.loc[1, 'location'] = ' '.join(location)
                ner = list()
                location = list()
                data_location_text.loc[1, 'ID'] = str(data.loc[i, 'ID'])
                data_location_text.loc[1, 'i'] = i
                data_location_text.loc[1, 'text'] = str(data.loc[i, 'text'])
                data_location_text.loc[1, 'translated'] = str(data.loc[i, 'translated'])
                data_location_text.loc[1, 'source_id'] = str(data.loc[i, 'source_id'])
                data_location_text.loc[1, 'date_time'] = str(data.loc[i, 'date_time'])
                data_location_text.loc[1, 'photo_video_id'] = str(data.loc[i, 'photo_video_id'])
                data_location_text.loc[1, 'photofilename'] = str(data.loc[i, 'photofilename'])                 
                data_location = pd.concat([data_location, data_location_text])
                data_location_text = pd.DataFrame(data = d, dtype = str)           
                print(i)
            else:
                print(i)
    except:
        data_location_text.loc[1, 'ner'] = 'check manually'
        data_location_text.loc[1, 'location'] = 'check manually'
        data_location_text.loc[1, 'ID'] = str(data.loc[i, 'ID'])
        data_location_text.loc[1, 'i'] = i
        data_location_text.loc[1, 'text'] = str(data.loc[i, 'text'])
        data_location_text.loc[1, 'translated'] = str(data.loc[i, 'translated'])
        data_location_text.loc[1, 'source_id'] = str(data.loc[i, 'source_id'])
        data_location_text.loc[1, 'date_time'] = str(data.loc[i, 'date_time'])
        data_location_text.loc[1, 'photo_video_id'] = str(data.loc[i, 'photo_video_id'])
        data_location_text.loc[1, 'photofilename'] = str(data.loc[i, 'photofilename'])                 
        data_location = pd.concat([data_location, data_location_text])
        data_location_text = pd.DataFrame(data = d, dtype = str)
        print(i)

            

data_location.to_csv('/home/stathis/Desktop/selective_replicate.csv')