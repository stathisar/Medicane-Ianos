import pandas as pd
import re
import emoji
from langdetect import detect
from translatepy import Translator
import datetime

df = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/Instagram_captions/Instagram_captions.csv')
df['cleanedCaption'] = ''
df['translationPerformed'] = ''
df['postedTime'] = ''
translator = Translator()

def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_emoji(text):
    """Remove emoticons from text."""
    return emoji.replace_emoji(text, replace = '')

def from_unix_time(timestamp_str):
    """Convert 'YYYY-MM-DD HH:MM:SS' in unix time."""
    dt = datetime.datetime.fromtimestamp(timestamp_str)
    dt = dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)


#translate
for i in range(0,len(df)):
	no_urls = remove_urls(df.loc[i, 'text'].replace('\n',''))
	df.loc[i, 'postedTime'] = from_unix_time(df.loc[i, 'timestamp'])
	try:
		lang = detect(no_urls)
	except:
		lang = 'el'
		df.loc[i, 'translationPerformed'] = 'detect lang error'
	if lang != 'el':
		l = translator.translate(str(no_urls), source_language = lang,  destination_language = 'el')
		l = remove_emoji(l.result.lower())
		df.loc[i,'cleanedCaption'] = str(l)
		df.loc[i, 'translationPerformed'] = 'translated successfully'
	else:
		df.loc[i,'cleanedCaption'] = str(no_urls).lower()
		df.loc[i,'cleanedCaption'] = remove_emoji(df.loc[i, 'cleanedCaption'])
		df.loc[i, 'translationPerformed'] = 'translation was not necessary'
		
		
		
df.to_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/Instagram_captions/Ins_captions_processed.csv')	

