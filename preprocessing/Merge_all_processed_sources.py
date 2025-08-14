import pandas as pd

x_dataset = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/X/X_data_processed.csv')

ins_captions = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/Instagram_captions/Ins_captions_processed.csv')
flickr_captions = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/Flickr_captions/flickr_processed.csv')
yt_metadata = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/YouTube_captions/YouTube_data_processed.csv')



#x_dataset columns: tweetText, id, createdAt, unixtime, cleanedTweet, translationPerformed
x_columns = ['tweetText', 'id', 'createdAt', 'unixtime', 'cleanedTweet', 'translationPerformed']
x_data_to_merge = x_dataset[x_columns]
x_data_to_merge['source'] = 'x'

#ins_captions columns: text, id, postedTime, timestamp, cleanedCaption, translationPerformed
ins_columns = ['text','id', 'postedTime', 'timestamp', 'cleanedCaption', 'translationPerformed']
ins_captions_to_merge = ins_captions[ins_columns]
ins_captions_to_merge['source'] = 'instagram'

#flickr_captions columns: Caption, Photo.ID, Date.Uploaded, unixtime, cleanedText, translationPerformed
flickr_columns = ['Caption', 'Photo.ID', 'Date.Uploaded', 'unixtime', 'cleanedText', 'translationPerformed']
flickr_captions_to_merge = flickr_captions[flickr_columns]
flickr_captions_to_merge['source'] = 'flickr'

#yt_metadata columns: Merged_Text, video_id, published_at, unixtime, Merged_translated_text, Merged_translationPerformed


yt_metadata['Merged_Text'] = ''
yt_metadata['Merged_translated_text'] = ''
yt_metadata['Merged_translationPerformed'] = ''

yt_metadata['Merged_Text'] = yt_metadata['title'] + ' ' + yt_metadata['description']
yt_metadata['Merged_translated_text'] = yt_metadata['cleanedYouTubeTitle'] + ' ' + yt_metadata['cleanedYouTubeDesc']
yt_metadata['Merged_translationPerformed'] = yt_metadata['translationPerformedTitle'] + ' ' + yt_metadata['translationPerformedDesc']

yt_columns = ['Merged_Text', 'video_id', 'published_at', 'unixtime', 'Merged_translated_text', 'Merged_translationPerformed']
yt_metadata_to_merge = yt_metadata[yt_columns]
yt_metadata_to_merge['source'] = 'YouTube'

#rename columns
x_data_to_merge.columns = ['text','id','datetime','unixtime','processed','translationPerformed','source']
ins_captions_to_merge.columns = ['text','id','datetime','unixtime','processed','translationPerformed','source']
flickr_captions_to_merge.columns = ['text','id','datetime','unixtime','processed','translationPerformed','source']
yt_metadata_to_merge.columns = ['text','id','datetime','unixtime','processed','translationPerformed','source']


#rbind
data_all = pd.concat([x_data_to_merge, ins_captions_to_merge, flickr_captions_to_merge, yt_metadata_to_merge])
