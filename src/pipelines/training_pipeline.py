import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTranformation
import os
import sys


if __name__ == '__main__':
    
    # Data Ingestion
    obj = DataIngestion()
    train_path,test_path = obj.initiate_ingestion()
    print(train_path,test_path)
    
    # Data Transformation
    data_transformation = DataTranformation()
    train_array,test_array,object_path = data_transformation.initiate_data_transformation(train_path,test_path)
    
    
    
    