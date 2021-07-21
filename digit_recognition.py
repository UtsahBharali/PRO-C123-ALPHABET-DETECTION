import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.ImageOps
import os,ssl,time 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image 

#setting an https 
if(not os.environ.get('PYTHONHTTPSVERFY','')and getattr(ssl,'_create_unverified_context',None)):
    ssl.create_default_context = ssl._create_unverified_context

#fetching the data
X,y = fetch_openml('mnist_784',version = 1,return_X_y = True)
print(pd.Series(y).value_counts())
classes  = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

#spliting the data and scaling it 
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 9,train_size = 7500,test_size=2500)

#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#fitting the data into the model