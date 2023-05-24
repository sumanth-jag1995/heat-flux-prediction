import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
    num_imputer_obj_file_path = os.path.join("artifacts","numimputer.pkl")
    cat_imputer_obj_file_path = os.path.join("artifacts","catimputer.pkl")
    label_encoder_obj_file_path = os.path.join("artifacts","label_encoder.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read of train and test as dataframe completed")

            # Define numerical & categorical features
            cat_features = train_df.select_dtypes(include = ['object']).columns.tolist()
            num_features = train_df.drop(['x_e_out [-]'], axis=1).select_dtypes(include = ['float64']).columns.tolist()
            
            # Handling missing values in numerical variables
            logging.info("instantiate numerical imputer object")
            num_imputer_obj = SimpleImputer(strategy='mean')
            train_df[num_features] = num_imputer_obj.fit_transform(train_df[num_features])
            test_df[num_features] = num_imputer_obj.transform(test_df[num_features])

            # Handling missing values in categorical variables
            logging.info("instantiate numerical imputer object")
            cat_imputer_obj = SimpleImputer(strategy='most_frequent')
            train_df[cat_features] = cat_imputer_obj.fit_transform(train_df[cat_features])
            test_df[cat_features] = cat_imputer_obj.transform(test_df[cat_features])

            # Label encoding for categorical variables
            logging.info("instantiate label encoder object")
            label_encoder_obj = dict()
            for col in cat_features:
                label_encoder_obj[col] = LabelEncoder()
                train_df[col] = label_encoder_obj[col].fit_transform(train_df[col].astype(str))
                test_df[col] = label_encoder_obj[col].transform(test_df[col].astype(str))

            # Preprocess column names as some regressors in the ensemble can't handle column names not having alphanumeric or underscore
            train_df.columns = [re.sub(r'[\[\]<>\s]+', '_', col) for col in train_df.columns]
            test_df.columns = [re.sub(r'[\[\]<>\s]+', '_', col) for col in test_df.columns]

            logging.info("instantiate preprocessing object")
            preprocessing_obj = StandardScaler()
            
            # Define columns to be dropped/selected
            target_column_name = "x_e_out_-_"
            id_column_name = "id"

            # Split the dataframe into input features and target feature
            input_feature_train_df = train_df.drop(columns=[target_column_name, id_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name, id_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on train and test dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate the input features and target features into a single numpy array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info(f"Saved numerical imputer object.")
            save_object(
                file_path = self.data_transformation_config.num_imputer_obj_file_path,
                obj = num_imputer_obj
            )

            logging.info(f"Saved categorical imputer object.")
            save_object(
                file_path = self.data_transformation_config.cat_imputer_obj_file_path,
                obj = cat_imputer_obj
            )

            logging.info(f"Saved label encoder object.")
            save_object(
                file_path = self.data_transformation_config.label_encoder_obj_file_path,
                obj = label_encoder_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.num_imputer_obj_file_path,
                self.data_transformation_config.cat_imputer_obj_file_path,
                self.data_transformation_config.label_encoder_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)