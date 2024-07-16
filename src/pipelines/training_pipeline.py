from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTranformation
from src.components.model_trainer import ModelTrainer

def trigger_pipeline():
    # Data Ingestion
    obj = DataIngestion()
    train_path,test_path = obj.initiate_ingestion()
    
    # Data Transformation
    data_transformation = DataTranformation()
    train_array,test_array,object_path = data_transformation.initiate_data_transformation(train_path,test_path)
    
    # Model Trainer 
    model_trainer = ModelTrainer()
    model_trainer.initiate_training(train_array,test_array)
    
if __name__ == '__main__':
    trigger_pipeline()