import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.cat_features = ['author', 'geometry']
        self.num_features = ['pressure_MPa_', 'mass_flux_kg/m2-s_',	'D_e_mm_', 'D_h_mm_', 'length_mm_', 'chf_exp_MW/m2_']

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            num_imputer_path = 'artifacts/numimputer.pkl'
            cat_imputer_path = 'artifacts/catimputer.pkl'
            label_encoder_path = 'artifacts/numimputer.pkl'
            feature_selector_path = 'artifacts/fselector.pkl'
            
            features.author = features.author.astype('object')
            features.geometry = features.geometry.astype('object')
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            num_imputer = load_object(file_path= num_imputer_path)
            cat_imputer = load_object(file_path= cat_imputer_path)
            label_encoder = load_object(file_path= label_encoder_path)
            feature_selector = load_object(file_path = feature_selector_path)

            features[self.num_features] = num_imputer.transform(features[self.num_features])
            features[self.cat_features] = cat_imputer.transform(features[self.cat_features])

            for col in self.cat_features:
                features[col] = label_encoder[col].transform(features[col].astype(str))
            
            data_scaled = preprocessor.transform(features)
            data_scaled_fs = feature_selector.transform(data_scaled)
            preds = model.predict(data_scaled_fs)
            return preds
        except Exception as e:
            raise CustomException(e, sys)        


class CustomData:
    def __init__(self,
                 author: str,
                 geometry: str,
                 pressure: float,
                 mass_flux: float,
                 d_e: float,
                 d_h: float,
                 length: float,
                 chf_exp: float):
        self.author = author
        self.geometry = geometry
        self.pressure = pressure
        self.mass_flux = mass_flux
        self.d_e = d_e
        self.d_h = d_h
        self.length = length
        self.chf_exp = chf_exp

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "author": [self.author],
                "geometry": [self.geometry],
                "pressure_MPa_": [self.pressure],
                "mass_flux_kg/m2-s_": [self.mass_flux],
                "D_e_mm_": [self.d_e],
                "D_h_mm_": [self.d_h],
                "length_mm_": [self.length],
                "chf_exp_MW/m2_": [self.chf_exp],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)