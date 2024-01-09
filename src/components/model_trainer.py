from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
import pickle
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass
import os
import sys
from src.utils import evaluate_model
@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_training(self,train_array,test_array):        
        try:
            logging.info('Splitting Dependent and Independent Features from Train and Test Array')
            X_train,X_test,y_train,y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'ElasticNet Regression': ElasticNet(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
            }
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models) 
            print('\n=========================================================================')
            logging.info(f'Model Report :{model_report}')
            
            # Getting the Model with best R2 Score value
            best_model_score = max(sorted(model_report.values())) 
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            print('\n==========================================================================')
            print(f'Best Model Found, Model Name: {best_model_name} R2 Score : {best_model_score}')
            print('\n==========================================================================') 
            logging.info(f'Best Model Found, Model Name: {best_model_name} R2 Score : {best_model_score}') 
            
            save_object(file_path=ModelTrainerConfig.trained_model_path,obj=best_model)
        except Exception as e:
            raise CustomException(e,sys)



