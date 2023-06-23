from src.entity import artifact_entity,config_entity
from src.exception import CustomException
from src.logger import logging
import os,sys 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sklearn
import pandas as pd

from src import utils
import numpy as np
from autoimpute.imputations import SingleImputer
from src.config import TARGET_COLUMN



class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            categorical_columns = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
             'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
            'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
            'stalk_surface_below_ring', 'stalk_color_above_ring',
             'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number',
             'ring_type', 'spore_print_color', 'population', 'habitat']
            simple_imputer = SingleImputer(copy=True, imp_kwgs=None, predictors='all', seed=None,
              strategy='categorical', visit='default')
            pipeline = Pipeline(steps=[
                    ('Imputer',simple_imputer),
                    ("one_hot_encoder",OneHotEncoder())
                ])
            preprocessor=ColumnTransformer(
                [
                ("cat_pipelines",pipeline,categorical_columns)
               ])

            return
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            # print(train_df.isnull().sum())
            # print(test_df.isnull().sum())
            # train_df.replace(to_replace="?",value=np.nan,inplace=True)
            # test_df.replace(to_replace="?",value=np.nan,inplace=True)
            # train_df["stalk_root"] =train_df["stalk_root"].replace('?',np.nan)
            # test_df["stalk_root"] =test_df["stalk_root"].replace('?',np.nan)
            # train_df.replace({"?",np.nan},inplace=True)
            # test_df.replace({"?",np.nan},inplace=True)

            # print(train_df.isnull().sum())
            # print(test_df.isnull().sum())
            
            #selecting input feature for train and test dataframe
            input_feature_train_df=train_df.drop(columns = [TARGET_COLUMN , 'index'],axis=1)
            input_feature_test_df=test_df.drop(columns = [TARGET_COLUMN , 'index'],axis=1)

            #selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            print(target_feature_test_arr)

            #simple_imputer = SingleImputer(strategy='categorical')
            # simple_imputer= SimpleImputer(strategy='most_frequent')
            # logging.info(f"simple imputer")
            # #simple_imputer.fit_transform(input_feature_train_df)
            # logging.info(f"simple imputer finish")

            # input_feature_train_df =simple_imputer.fit_transform(input_feature_train_df)
            # input_feature_test_df= simple_imputer.fit_transform(input_feature_test_df)


            one_hot_encoder = OneHotEncoder(drop='first',handle_unknown='ignore')
            one_hot_encoder.fit(input_feature_train_df)
            logging.info(f"simple imputer")
            #transforming input features
            input_feature_train_arr = one_hot_encoder.transform(input_feature_train_df).toarray()
            input_feature_test_arr = one_hot_encoder.transform(input_feature_test_df).toarray()
            

            # smt = SMOTETomek(random_state=42)
            # logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            # input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            # logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            
            # logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            # input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            # logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=one_hot_encoder)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=label_encoder)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)