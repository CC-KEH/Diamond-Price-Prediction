import os
import sys
import numpy as np 
import pandas as pd 
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in models:
            model = models[i]
            model.fit(X_train,y_train)
            
            # Predictions
            preds = model.predict(X_test)
            
            # R2 Score
            r2 = r2_score(preds,y_test)
            
            report[i] = r2
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
      


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
