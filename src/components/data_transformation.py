from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np 
import pandas as pd
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import sys
import os
from dataclasses import dataclass 

@dataclass 
class DataTransformerConfig(self):
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTranformation:

    def __init__(self):
        self.data_transformation_config = DataTransformerConfig()
    def get_data_tranformation_object(self):
        try:
            loggin.info('Data Transformation Initiated')

            # categorical_cols and numerical_cols
            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']
            
            # Custom Ranking
            cut_ranking = ['Fair','Good','Very Good','Premium','Ideal']
            color_ranking = ['J','I','H','G','F''E','D']
            clarity_ranking = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Data Tranformation pipeline Initiated')

            num_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OrdinalEncoder(categories=[cut_ranking,color_ranking,clarity_ranking])),
                ('scaler',StandardScaler())
            ])
            
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            logging.info('Data Tranformation Completed')

            return preprocessor
            
        except Exception as e:
            logging.info('Exception occured in Data Transformation')
            raise CustomException(e,sys)

            
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read Train and Test data completed')
            logging.info(f'Train data: \n {train_df.head().to_string()}')
            logging.info(f'Test data: \n {test_df.head().to_string()}')

            loggin.info('Obtaining preprocessor object')

            preprocessor_obj = self.get_data_tranformation_object()

            target_col = 'price'
            drop_col = [target_col,'id','depth','x','y','z']

            # Dividing dataset into dependent and independent features
            input_feature_train_df = train_df.drop(columns=drop_col,axis=1)
            target_feature_train_df = train_df[target_col]
            
            # Test data 
            input_feature_test_df = test_df.drop(columns=drop_col,axis=1)
            target_feature_test_df = test_df[target_col]
            
            # Data Tranformation
            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)

            train_array = np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array,np.array(target_feature_test_df)]
            
            logging.info('Transformed Data Successfully')


            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj)

            return(train_array,test_array,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)

