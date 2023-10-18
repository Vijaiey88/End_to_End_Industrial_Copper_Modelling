import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


# Define a custom transformer for mapping categorical features
class CategoricalMapper:
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X):
        return X

    def fit(self, X, y=None):
        return self

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformation(self, df):
        try:
            # Step 1
            df['item_date'] = df['item_date'].replace(19950000.0, 19950101.0)

            # Step 2
            df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce')
            df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce')

            # Step 3
            df['duration'] = abs(df['item_date'] - df['delivery date'])
            df.drop(columns='delivery date', axis=1, inplace=True)

            # Step 4
            df['quantity tons'] = df['quantity tons'].replace('e', np.nan)

            # Step 5
            df['quantity tons'] = pd.to_numeric(df['quantity tons'])

            # Step 6
            columns_to_fill_with_mode = ['item_date', 'customer', 'status', 'application', 'country']
            columns_to_fill_with_median = ['thickness', 'quantity tons', 'selling_price', 'duration']

            for column in columns_to_fill_with_mode:
                df[column] = df[column].fillna(df[column].mode()[0])

            for column in columns_to_fill_with_median:
                df[column] = df[column].fillna(df[column].median())

            # Step 7
            df['material_ref'] = df['material_ref'].str.lstrip('0')
            def impute_nan_material(df, feature):
                random_sample = df[feature].dropna().sample(df[feature].isnull().sum(), random_state=0)
                random_sample.index = df[df[feature].isnull()].index
                df.loc[df[feature].isnull(), feature] = random_sample

            impute_nan_material(df, 'material_ref')

            # Step 8
            q1 = df['quantity tons'].quantile(0.25)
            upper_bound = df['quantity tons'].quantile(0.75)
            df['quantity tons'] = np.where(df['quantity tons'] < -1000, q1, df['quantity tons'])
            df['quantity tons'] = np.where(df['quantity tons'] < 0, df['quantity tons'].abs(), df['quantity tons'])
            df['quantity tons'] = np.where(df['quantity tons'] > 1.000000e+05, upper_bound, df['quantity tons'])
            df['quantity tons'] = np.log(df['quantity tons'])

            # Step 9
            upper_bound = 250
            df['thickness'] = np.where(df['thickness'] > upper_bound, upper_bound, df['thickness'])
            df['thickness'] = np.log(df['thickness'])

            # Step 10
            q1 = df['selling_price'].quantile(0.25)
            upper_bound = df['selling_price'].quantile(0.75)
            df['selling_price'] = np.where(df['selling_price'] <= 0, q1, df['selling_price'])
            df['selling_price'] = np.where(df['selling_price'] > 1e4, upper_bound, df['selling_price'])
            df['selling_price'] = df['selling_price']**(1/2)

            # Step 11
            for feature in ['status', 'item type', 'material_ref']:
                feature_map = df[feature].value_counts().to_dict()
                df[feature] = df[feature].map(feature_map)

            # Step 12
            df['item_date'] = df['item_date'].astype(str)
            df['item_delivery_year'] = df['item_date'].apply(lambda x: x.split('-')[0])
            df['item_delivery_month'] = df['item_date'].apply(lambda x: x.split('-')[1])
            df['item_delivery_date'] = df['item_date'].apply(lambda x: x.split('-')[2])
            df.drop(columns='item_date', axis=1, inplace=True)
            df['item_delivery_year'] = df['item_delivery_year'].astype(int)
            df['item_delivery_month'] = df['item_delivery_month'].astype(int)
            df['item_delivery_date'] = df['item_delivery_date'].astype(int)

            # Create a pipeline for numerical columns
            numerical_pipeline = Pipeline([
                ('quantity tons', FunctionTransformer(lambda X: np.where(X < -1000, q1, X), validate=False)),
                ('thickness', FunctionTransformer(lambda X: np.log(np.where(X > 250, 250, X)), validate=False)),
                ('selling_price', FunctionTransformer(lambda X: X**(1/2), validate=False)),
            ])

            # Create a pipeline for categorical columns
            categorical_pipeline = Pipeline([
                ('map_categorical_features', CategoricalMapper(columns=['status', 'item type', 'material_ref'])),
            ])

            # Apply transformations to the corresponding columns
            numeric_columns = ['quantity tons', 'thickness', 'selling_price']
            categorical_columns = ['status', 'item type', 'material_ref']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numeric_columns),
                    ('cat', categorical_pipeline, categorical_columns),
                ],
                remainder='passthrough'
            )

            # Apply the transformations to the DataFrame
            transformed_data = preprocessor.fit_transform(df)

            return transformed_data

        except Exception as e:
            raise CustomException(e)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read train and test data completed.')
            logging.info('Obtaining preprocessing object')
            # Load the preprocessing object that you saved earlier
            preprocessing_obj = load_object(self.data_transformation_config.preprocessor_obj_file_path)

            # Use the preprocessing_obj to transform the input features
            input_feature_train_arr = preprocessing_obj.transform(train_df)
            input_feature_test_arr = preprocessing_obj.transform(test_df)


            train_arr = np.c_[
                input_feature_train_arr,
                np.array(train_df['selling_price'])
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(test_df['selling_price'])
            ]


            logging.info("Loaded and applied preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
