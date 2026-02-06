import pandas as pd
import os
from mlProject import logger
from catboost import CatBoostRegressor
from src.mlProject.config.configuration import ModelTrainerConfig
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        cbr = CatBoostRegressor(depth=self.config.depth, learning_rate=self.config.learning_rate,
                                iterations=self.config.iterations,verbose=self.config.verbose,
                                random_state=self.config.random_state)
        cbr.fit(train_x, train_y)

        joblib.dump(cbr, os.path.join(self.config.root_dir, self.config.model_name))