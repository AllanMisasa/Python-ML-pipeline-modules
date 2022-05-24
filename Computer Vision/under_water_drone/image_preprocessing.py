import urllib.request as request
import cv2 as cv
import numpy as np 
from image_fetch import get_url_paths, url_strip 

url = "https://github.com/AllanMisasa/Python-ML-pipeline-modules/blob/main/Computer%20Vision/DD_dataset"
ext = '.jpg'
image_paths = [url_strip(url) for url in get_url_paths(url, ext)] # Gets the paths of the images from the url

def get_image(number): # Gets the image from the url
    req = request.urlopen(image_paths[number]) # Opens the url
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8) # Converts the url to an array
    img = cv.imdecode(arr, -1) # Decodes the array
    return img # Returns the image

image = get_image(3) # Gets the image

def preprocess_image(image): # Preprocesses the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Converts the image to grayscale
    img_blur = cv.GaussianBlur(gray, (3, 3), 0) # Blurs the image
    dimensions = np.shape(gray) # The shape of the image
    cropped_image = gray[80:dimensions[1], 150:330] # Crops the image
    # img_thresh = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2) # Thresholds the image
    sobelx = cv.Sobel(img_blur, cv.CV_64F, 1, 0, ksize=5) # Applies the sobel filter to the x axis of the image
    sobely = cv.Sobel(img_blur, cv.CV_64F, 0, 1, ksize=5)  # Applies the sobel filter to the y axis of the image
    sobelxy = cv.Sobel(img_blur, cv.CV_64F, 1, 1, ksize=5) # Applies the sobel filter to the x and y axis of the image
    edges = cv.Canny(img_blur, 100, 200) # Applies the canny filter to the image
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY) # Thresholds the image
    return cropped_image

processed_image = preprocess_image(image)
    
cv.imshow('image', processed_image) # Shows the image
cv.waitKey(0) # Waits for a key press
cv.destroyAllWindows() # Destroys all the windows
