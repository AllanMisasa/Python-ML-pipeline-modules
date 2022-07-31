import urllib.request as request # To fetch images from repo
import cv2 as cv # Main image processing module
import numpy as np # Main container for dataset
import pandas as pd # Intermediary containter for dataset
import matplotlib.pyplot as plt # Main 2D visualization library
# from sklearnex import patch_sklearn # To speed up most SkLearn classifiers - MUST be loaded and instantiated before importing sklearn
# patch_sklearn()
from sklearn.model_selection import train_test_split, StratifiedKFold # For dataset splitting and crossvalidation.
# Must be stratified as there is a class imbalance.
from sklearn.linear_model import LogisticRegression # For regressors
from sklearn.preprocessing import StandardScaler # To scale dataset
from sklearn.metrics import confusion_matrix, RocCurveDisplay, auc # Different metrics for analyzing ML performance
from sklearn.ensemble import RandomForestClassifier # Random forest classifier

def get_image(image_path): # Gets the image from the url
    req = request.urlopen(image_path) # Opens the url
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8) # Converts the url pip to an array
    img = cv.imdecode(arr, -1) # Decodes the array
    return img # Returns the image

def preprocess_image(image): # Preprocesses the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Converts the image to grayscale
    contrasted = cv.equalizeHist(gray)
    img_blur = cv.GaussianBlur(contrasted, (3, 3), 0) # Blurs the image
    sobelx = cv.Sobel(img_blur, cv.CV_64F, 1, 0, ksize=5) # Applies the sobel filter to the x axis of the imagepip
    sobely = cv.Sobel(img_blur, cv.CV_64F, 0, 1, ksize=5)  # Applies the sobel filter to the y axis of the image
    sobelxy = cv.Sobel(img_blur, cv.CV_64F, 1, 1, ksize=5) # Applies the sobel filter to the x and y axis of the image
    edges = cv.Canny(img_blur, 100, 200) # Applies the canny filter to the image
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY) # Thresholds the image
    return img_blur

def load_unique_images(): # Loads the unique images
    df = pd.read_excel('Computer Vision/under_water_drone/sea_samples_file_names_targets.xlsx').drop_duplicates(subset=['unique_id']) # Reads the excel file
    url = "https://raw.githubusercontent.com/AllanMisasa/Python-ML-pipeline-modules/main/Computer%20Vision/DD_dataset/transformed/"
    df['file_name'] = url + df['file_name'] # Adds the url to the image paths
    return(df) # Returns the unique images

'''
First preprocesses the images with chosen parameters. Then reshapes the images to 1D so they can be fed to traditional regression models.
Displays and example image - mostly for checking. Class balance is displayed. 
Scales and transforms dataset to limit the effect of extreme values.
Returns train and test sets.
'''

def generate_dataset(dataframe):
    image_data = [get_image(image) for image in dataframe['file_name']] # Gets the images from the url
    image_data = [preprocess_image(image) for image in image_data] # Preprocesses the images
    image_data = np.array(image_data) # Converts the images to a numpy array
    image_data = image_data.reshape(image_data.shape[0], -1) # Reshapes the image dataframe to a 2D array
    targets = dataframe['target'] # Gets the targets
    print(image_data[1].shape)
    print("Example 1D image: ", image_data[0])
    print("Class balance is [unacceptable samples | acceptable samples]: ", np.bincount(targets)) # Prints the class balance
    X_train, X_test, y_train, y_test = train_test_split(image_data, targets, test_size=0.2, random_state=42) # Splits the data into training and testing sets
    scaler = StandardScaler() # Creates a scaler
    X_train = scaler.fit_transform(X_train) # Fits and transforms the training data
    X_test = scaler.transform(X_test) # Transforms the testing data
    return X_train, X_test, y_train, y_test

def classification(classifier, X_train, X_test, y_train, y_test, cross_val = True): # Logistic classification
    classifier.fit(X_train, y_train) # Fits the model to the training data
    y_pred = classifier.predict(X_test) # Predicts the testing data
    print("Logistic classification accuracy: ", classifier.score(X_test, y_test)) # Prints the accuracy of the model
    print("Confusion matrix: ", confusion_matrix(y_test, y_pred)) # Prints the confusion matrix
    if cross_val == True:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        cv = StratifiedKFold(n_splits=5) # Prepare cross-validation and select number of splits
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        for i, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            viz = RocCurveDisplay.from_estimator(
                classifier, X[test], y[test], name="ROC fold {}".format(i),
                alpha=0.3, lw=1, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="Receiver operating characteristic example",
        )
        ax.legend(loc="lower right")
        plt.show()


df = load_unique_images() # Loads the unique images
X_train, X_test, y_train, y_test = generate_dataset(df) # Generates the dataset
logreg = LogisticRegression()
ranfor = RandomForestClassifier(random_state=0)
classification(logreg, X_train, X_test, y_train, y_test) # Logistic classification


'''
cv.imshow('image', image_data[0]) # Shows the first image
cv.waitKey(0) # Waits for the user to press a key
cv.destroyAllWindows() # Closes all the windows


processed_image = preprocess_image(image)
    
cv.imshow('image', processed_image) # Shows the image
cv.waitKey(0) # Waits for a key press
cv.destroyAllWindows() # Destroys all the windows
'''