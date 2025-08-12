import pandas as pd
import re
import emoji
from langdetect import detect
from translatepy import Translator
from datetime import datetime


df = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/YouTube_captions/youtube_videos_all_march_@.csv')
df['cleanedYouTubeTitle'] = ''
df['cleanedYouTubeDesc'] = ''
df['translationPerformedTitle'] = ''
df['translationPerformedDesc'] = ''
df['unixtime'] = ''

translator = Translator()

def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_emoji(text):
    """Remove emoticons from text."""
    return emoji.replace_emoji(text)

def to_unix_time(timestamp_str):
    """Convert 'YYYY-MM-DD HH:MM:SS' in unix time."""
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp())


#translate
for i in range(0,len(df)):
	no_urls_title = remove_urls(str(df.loc[i, 'title']).replace('\n',''))
	no_urls_desc = remove_urls(str(df.loc[i, 'description']).replace('\n',''))
	df.loc[i, 'unixtime'] = to_unix_time(df.loc[i, 'published_at'].replace('T', ' ').replace('Z',''))
	try:
		lang_title = detect(no_urls_title)
	except:
		lang_title = 'el'
		df.loc[i, 'translationPerformedTitle'] = 'detect lang error'
	try:
		lang_desc = detect(no_urls_desc)
	except:
		lang_desc = 'el'
		df.loc[i, 'translationPerformedDesc'] = 'detect lang error'
	if lang_title != 'el':
		l = translator.translate(str(no_urls_title), source_language = lang_title,  destination_language = 'el')
		l = remove_emoji(l.result.lower())
		df.loc[i,'cleanedYouTubeTitle'] = str(l)
		df.loc[i, 'translationPerformedTitle'] = 'translated successfully'
	else:
		df.loc[i,'cleanedYouTubeTitle'] = str(no_urls_title).lower()
		df.loc[i,'cleanedYouTubeTitle'] = remove_emoji(df.loc[i, 'cleanedYouTubeTitle'])
		df.loc[i, 'translationPerformedTitle'] = 'translation was not necessary'
	if lang_desc != 'el':
		l = translator.translate(str(no_urls_desc), source_language = lang_desc,  destination_language = 'el')
		l = remove_emoji(l.result.lower())
		df.loc[i,'cleanedYouTubeDesc'] = str(l)
		df.loc[i, 'translationPerformed'] = 'translated successfully'
	else:
		df.loc[i,'cleanedYouTubeDesc'] = str(no_urls_desc).lower()
		df.loc[i,'cleanedYouTubeDesc'] = remove_emoji(df.loc[i, 'cleanedYouTubeDesc'])
		df.loc[i, 'translationPerformed'] = 'translation was not necessary'
		
		
		
df.to_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/YouTube_captions/YouTube_data_processed.csv')	

