import flickrapi
import csv
import time
from typing import Optional
from datetime import datetime

# Add the corresponding API keys
API_KEY = 'f0e5e766faf72869d9ec887e882e5b55'
API_SECRET = '6117f46de7ed60fb'

# Creation of Flickr API client
flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET, format='parsed-json')


def date_to_unix(date_str: Optional[str]) -> Optional[int]:
    if date_str:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return int(dt.timestamp())
    return None
    
    
def scrape_flickr(
    tag: str,
    max_pages: int = 5,
    per_page: int = 100,
    output_csv: str = 'flickr_photos.csv',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
    ):
    min_taken_date = date_to_unix(start_date)
    max_taken_date = date_to_unix(end_date)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'title', 'url', 'tags', 'date_taken']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for page in range(1, max_pages + 1):
            try:
                print(f'Fetching page {page} for tag "{tag}"...')
                photos = flickr.photos.search(tags=tag, per_page=per_page, page=page, 	extras='tags,date_taken,url_o,url_c,url_l,url_m', min_taken_date = min_taken_date, max_taken_date = max_taken_date)
                photo_list = photos['photos']['photo']
                if not photo_list:
                    print('No more photos found.')
                    break
                for photo in photo_list:
                    # Trying to find the most optimum url
                    url = photo.get('url_o') or photo.get('url_l') or photo.get('url_c') or photo.get('url_m') or ''
                    writer.writerow({
                        'id': photo['id'],
                        'title': photo.get('title', ''),
                        'url': url,
                        'tags': photo.get('tags', ''),
                        'date_taken': photo.get('date_taken', '')
                    })
                time.sleep(1)  # For not overchallenge the API
            except Exception as e:
                print(f'Error fetching page {page}: {e}')
                break

#example
if __name__ == "__main__":
    scrape_flickr('nature', max_pages=3, start_date='2020-09-10', end_date='2020-10-30')
    
#example
#scrape_flickr('nature', max_pages=3, per_page=3, start_date = '2020-09-10', end_date = '2020-10-30') #custom max_pages and per_page
