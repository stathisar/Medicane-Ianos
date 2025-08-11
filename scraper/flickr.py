import flickrapi
import csv
import time

# Add your API keys
API_KEY = 'APISTRING'
API_SECRET = 'APISECRET'

# Creation of Flickr API client
flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET, format='parsed-json')

def scrape_flickr(tag, max_pages=5, per_page=100, output_csv='flickr_photos.csv'):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'title', 'url', 'tags', 'date_taken']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for page in range(1, max_pages + 1):
            try:
                print(f'Fetching page {page} for tag "{tag}"...')
                photos = flickr.photos.search(tags=tag, per_page=per_page, page=page, extras='tags,date_taken,url_o,url_c,url_l,url_m')
                photo_list = photos['photos']['photo']
                if not photo_list:
                    print('No more photos found.')
                    break
                for photo in photo_list:
                    # Finding the most optimum URL.
                    url = photo.get('url_o') or photo.get('url_l') or photo.get('url_c') or photo.get('url_m') or ''
                    writer.writerow({
                        'id': photo['id'],
                        'title': photo.get('title', ''),
                        'url': url,
                        'tags': photo.get('tags', ''),
                        'date_taken': photo.get('date_taken', '')
                    })
                time.sleep(1)  # Για να μην ζορίσουμε το API πολύ
            except Exception as e:
                print(f'Error fetching page {page}: {e}')
                break

#example, usage 1    
if __name__ == "__main__":
    # Example: Searching for photos, using the tag #nature.
    scrape_flickr('nature', max_pages=3)
    
#example, usage 2
scrape_flickr('nature', max_pages=3, per_page=3) #custom max_pages and per_page
