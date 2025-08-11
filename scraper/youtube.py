import os
from googleapiclient.discovery import build
from datetime import datetime
import csv

#Add your API Key
API_KEY = 'ADD-API-KEY'

youtube = build('youtube', 'v3', developerKey=API_KEY)

def search_youtube(query, published_after=None, max_results=50):
    """
    Query based video search, date time filtering
    """
    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        order='date',  # most recent first
        publishedAfter=published_after,  # e.g.'2023-01-01T00:00:00Z'
        maxResults=max_results
    )
    response = request.execute()
    videos = []
    for item in response.get('items', []):
        video_data = {
            'videoId': item['id']['videoId'],
            'title': item['snippet']['title'],
            'description': item['snippet']['description'],
            'publishedAt': item['snippet']['publishedAt'],
            'channelTitle': item['snippet']['channelTitle']
        }
        videos.append(video_data)
    return videos

def export_to_csv(videos, filename='youtube_results.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['videoId', 'title', 'description', 'publishedAt', 'channelTitle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for video in videos:
            writer.writerow(video)
            

if __name__ == '__main__':
    query = "rain"
    published_after = "2024-01-01T00:00:00Z"
    results = search_youtube(query=query, published_after=published_after, max_results=3)
    export_to_csv(results)
    print(f"{len(results)} videos exported in youtube_results.csv")
