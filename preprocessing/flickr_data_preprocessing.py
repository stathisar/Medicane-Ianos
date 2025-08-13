import pandas as pd
import re
import emoji
from langdetect import detect
from translatepy import Translator
from datetime import datetime


df = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/Flickr_captions/flickr_metadata_translated_captions_probably_referred_to_manuscript.csv')
df['cleanedText'] = ''
df['translationPerformed'] = ''
translator = Translator()

def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_emoji(text):
    """Remove emoticons from text."""
    return emoji.replace_emoji(text, replace = '')

def to_unix_time(timestamp_str):
    """Convert 'YYYY-MM-DD HH:MM:SS' in unix time."""
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp())


#translate
for i in range(0,len(df)):
	no_urls = remove_urls(df.loc[i, 'Caption'].replace('\n',''))
	df.loc[i, 'unixtime'] = to_unix_time(df.loc[i, 'Date.Uploaded'])
	try:
		lang = detect(no_urls)
	except:
		lang = 'el'
		df.loc[i, 'translationPerformed'] = 'detect lang error'
	if lang != 'el':
		try:
			l = translator.translate(str(no_urls), source_language = lang,  destination_language = 'el')
			l = remove_emoji(l.result.lower())
			df.loc[i,'cleanedText'] = str(l)
			df.loc[i, 'translationPerformed'] = 'translated successfully'
		except:
			df.loc[i, 'translationPermormed'] = 'translation error'		
	else:
		df.loc[i,'cleanedText'] = str(no_urls).lower()
		df.loc[i,'cleanedText'] = remove_emoji(df.loc[i, 'cleanedText'])
		df.loc[i, 'translationPerformed'] = 'translation was not necessary'
		
		
		
df.to_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/Flickr_captions/flickr_processed.csv')	

