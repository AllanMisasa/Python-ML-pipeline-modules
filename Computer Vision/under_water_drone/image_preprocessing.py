import urllib.request as request
import cv2 as cv
import numpy as np 
import pandas as pd

def get_image(number, image_paths): # Gets the image from the url
    req = request.urlopen(image_paths[number]) # Opens the url
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8) # Converts the url pip to an array
    img = cv.imdecode(arr, -1) # Decodes the array
    return img # Returns the image

def preprocess_image(image): # Preprocesses the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Converts the image to grayscale
    img_blur = cv.GaussianBlur(gray, (3, 3), 0) # Blurs the image
    sobelx = cv.Sobel(img_blur, cv.CV_64F, 1, 0, ksize=5) # Applies the sobel filter to the x axis of the image
    sobely = cv.Sobel(img_blur, cv.CV_64F, 0, 1, ksize=5)  # Applies the sobel filter to the y axis of the image
    sobelxy = cv.Sobel(img_blur, cv.CV_64F, 1, 1, ksize=5) # Applies the sobel filter to the x and y axis of the image
    edges = cv.Canny(img_blur, 100, 200) # Applies the canny filter to the image
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY) # Thresholds the image
    return gray

def load_unique_images(): # Loads the unique images
    df = pd.read_excel('Computer Vision/under_water_drone/sea_samples_file_names_targets.xlsx').drop_duplicates(subset=['unique_id']) # Reads the excel file
    url = "https://raw.githubusercontent.com/AllanMisasa/Python-ML-pipeline-modules/main/Computer%20Vision/DD_dataset/transformed/"
    df['file_name'] = url + df['file_name'] # Adds the url to the image paths
    return(df) # Returns the unique images

df = load_unique_images() # Loads the unique images

print(df.head())

'''
processed_image = preprocess_image(image)
    
cv.imshow('image', processed_image) # Shows the image
cv.waitKey(0) # Waits for a key press
cv.destroyAllWindows() # Destroys all the windows
'''