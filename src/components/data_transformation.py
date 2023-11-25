import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig():
    data_transformer = os.path.join('artifacts', 'data_transformer.pkl')


class DataTransformer():
    def __init__(self):
        self.ingestion_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender',
                                    'race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch',
                                    'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[(
                    'imputer', SimpleImputer(strategy='median')
                ), (
                    'scaler', StandardScaler()
                )]
            )

            cat_pipeline = Pipeline(
                steps=[(
                    'imputer', SimpleImputer(strategy='most_frequent')
                ), (
                    'one_hot_encoder', OneHotEncoder()
                )]
            )

            logging.info("Numerical columns Encoding Completed...")
            logging.info("Categorical columns Encoding Completed...")

            preprocessor = ColumnTransformer([
                ("numerical_pipeline", num_pipeline, numerical_features)
                ("categorical_pipeline", cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded Train set...")
            logging.info("Loaded Test set...")

            preprocessor = self.get_data_transformer_object()

            target_column = train_df.drop("math_score", axis=1)
            pass
        except:
            pass
