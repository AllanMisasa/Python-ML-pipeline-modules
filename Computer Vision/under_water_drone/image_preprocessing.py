import urllib.request as request
import cv2 as cv
import numpy as np 
import pandas as pd
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def get_image(image_path): # Gets the image from the url
    req = request.urlopen(image_path) # Opens the url
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

def generate_dataset(dataframe):
    image_data = [get_image(image) for image in dataframe['file_name']] # Gets the images from the url
    image_data = [preprocess_image(image) for image in image_data] # Preprocesses the images
    image_data = np.array(image_data) # Converts the images to a numpy array
    image_data = image_data.reshape(image_data.shape[0], -1) # Reshapes the images to a 2D array
    targets = dataframe['target'] # Gets the targets
    print("Class balance is [unacceptable samples | acceptable samples]: ", np.bincount(targets)) # Prints the class balance
    X_train, X_test, y_train, y_test = train_test_split(image_data, targets, test_size=0.2, random_state=42) # Splits the data into training and testing sets
    scaler = StandardScaler() # Creates a scaler
    X_train = scaler.fit_transform(X_train) # Fits and transforms the training data
    X_test = scaler.transform(X_test) # Transforms the testing data
    return X_train, X_test, y_train, y_test

def logistic_classification(X_train, X_test, y_train, y_test): # Logistic classification
    logreg = LogisticRegression() # Creates a logistic regression model
    logreg.fit(X_train, y_train) # Fits the model to the training data
    y_pred = logreg.predict(X_test) # Predicts the testing data
    print("Logistic classification accuracy: ", logreg.score(X_test, y_test)) # Prints the accuracy of the model

df = load_unique_images() # Loads the unique images
X_train, X_test, y_train, y_test = generate_dataset(df) # Generates the dataset
logistic_classification(X_train, X_test, y_train, y_test) # Logistic classification


'''
cv.imshow('image', image_data[0]) # Shows the first image
cv.waitKey(0) # Waits for the user to press a key
cv.destroyAllWindows() # Closes all the windows


processed_image = preprocess_image(image)
    
cv.imshow('image', processed_image) # Shows the image
cv.waitKey(0) # Waits for a key press
cv.destroyAllWindows() # Destroys all the windows
'''