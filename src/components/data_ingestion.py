import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset
            df = pd.read_csv("artifacts/data.csv")
            logging.info("Read the dataset as dataframe")

            # Create directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Filter for rows where target feature is not null
            filtered_df = df[df['x_e_out [-]'].isnull()!=True]
            filtered_df.reset_index(inplace=True,drop=True)

            logging.info("Train test split initiated")
            # Split the data into train and test set
            train_set, test_set = train_test_split(filtered_df, test_size = 0.2, random_state = 42)

            # Save the train data to csv
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            # Save the test data to csv
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _, _, _, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    modeltrainer = ModelTrainer(train_array, test_array)
    print(modeltrainer.initiate_model_trainer())