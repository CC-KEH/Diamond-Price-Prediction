import pandas as pd                                             
from sklearn.model_selection import train_test_split            
from src.logger import logging                                  
from src.exception import CustomException                       
from src.components.data_ingestion import DataIngestion         
from src.components.data_transformation import DataTranformation
from src.components.model_trainer import ModelTrainer           
from src.utils import load_object
import os                                                       
import sys                                                      

class PredictionPipeline:
    
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            print("Inside Prediction Pipeline")
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl') 
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            data_scaled = preprocessor.transform(features)
            
            pred = model.predict(data_scaled)
            
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)
          
                                                        
class CustomData:
    def __init__(self,carat,depth,table,x,y,z,cut,color,clarity):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict  = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            raise CustomException(e,sys)    