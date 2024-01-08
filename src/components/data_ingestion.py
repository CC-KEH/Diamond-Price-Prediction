import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import os
import sys


@dataclass
class DataIngestionConfig():
    train_path:str = os.path.join('artifacts','train.csv')
    test_path:str = os.path.join('artifacts','test.csv')
    raw_path:str = os.path.join('artifacts','raw.csv')
    
class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_ingestion(self):
        logging.info('Ingestion Initiated')
        
        try:
            df=pd.read_csv(os.path.join('notebooks/data/','gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path,index=False)
            
            logging.info('Raw data is created')
            
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_path,index=False,header=True)
            
            logging.info('Ingestion Completed')
            
            return(
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage')
            raise CustomException(e,sys)