# Alphabet-Recognition C122
 * We wrote a prediction algorithm which would detect the numbers from the given images.

 ## Used
  * Using data from ML library
  * Logistic regression
  * Confusion matrix

## Steps To Create A Prediction Algorithm To Detect Alphabets/Numbers
  * *We imported all the necessary libraries.*
  * *We fetched the data from the open ml library.*
  * *We printed the data to see what the images looked like.*
  * *We printed the number of images and the number of pixels.We also checked the array of the particular image.*
  * *We fit the tested and trained data to the model using logistic regression.*
  * *We made a prediction and checked the accuracy of the model.*
  * *We also created the confusion matrix.*

## More About Libraries
* *`import cv2` - This is the library with which we are going to use our computer's camera.*
* *`import numpy as np` - This is so that we can perform complex mathematical/list operations*
* *`import pandas as pd` - This is so that we can treat our data as DataFrames. We already know how helpful they are.*
* *`import seaborn as sns` - This is a python module to prettify the charts that we draw with `matplotlib`. We have used it a couple of times.*
* *`import matplotlib.pyplot as plt` - This library is used to draw the charts.*
* *`from sklearn.datasets import fetch_openml` - This function allows us to retrieve a data set by name from OpenML, a public repository for machine learning data and experiments.*
* *`from sklearn.model_selection import train_test_split` - This is to split our data into training and testing.*
* *`from sklearn.linear_model import LogisticRegression` - This is for creating a LogiticRegression Classifier.*
* *`from sklearn.metrics import accuracy_score` - This is to measure the accuracy score of the model.*

### Fact
* *Generally, there is a solver involved in all the logistic regressions, and the default solver is `liblinear`, which is highly efficient for linear logistic regression.*
* *This is also efficient with binary logistic regressions that we learnt earlier.*
* *For `multinomial` logistic regression, `solver='saga'` is highly efficient. It works well with large number of samples and supports `multinomial` logistic regressions, like this one.*
