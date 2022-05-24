import requests
from bs4 import BeautifulSoup


# url2 = 'https://raw.githubusercontent.com/AllanMisasa/Python-ML-pipeline-modules/main/Computer%20Vision/DD_dataset/'
url = "https://github.com/AllanMisasa/Python-ML-pipeline-modules/blob/main/Computer%20Vision/DD_dataset" # url for the images

def get_url_paths(url, ext='', params={}): # Gets the paths of the images from the url
    response = requests.get(url, params=params) # Gets the html from the url
    if response.ok: # If the html is ok
        response_text = response.text # Gets the text from the html
    else: # If the html is not ok
        return response.raise_for_status() # Raise an error
    soup = BeautifulSoup(response_text, 'html.parser') # Parses the html
    parent = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)] # Gets the links from the html
    return parent # Returns the links

ext = '.jpg' # The extension of the images

def url_strip(url): # Strips the url of the image
    start = 18 
    stop = 96
    first = url[:start] # Gets the first part of the url
    second = url[stop:] # Gets the second part of the url
    combined = first + second # Combines the two parts
    combined = combined.replace('blob/', '') # Removes the blob/ from the url
    combined = combined.replace('https://github.com/', 'https://raw.githubusercontent.com/') # Replaces the github url with the raw github url
    return combined # Returns the combined url

image_paths = [url_strip(url) for url in get_url_paths(url, ext)] # Gets the paths of the images from the url

#image_paths = get_url_paths(url2, ext)
print((image_paths[0])) # Prints the first image path

# example_image = Image.open(urllib.request.urlopen(image_paths[3])) # Opens the first image
# gray = ImageOps.grayscale(example_image) # Converts the image to grayscale
# gray.show() # Shows the image
