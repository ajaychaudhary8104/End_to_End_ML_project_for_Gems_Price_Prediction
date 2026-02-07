import os
from mlProject import logger
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def preprocessing(self):
        """
        Main preprocessing function that handles:
        - Loading data
        - Dropping irrelevant columns
        - Handling missing values
        - Encoding categorical variables
        - Scaling numerical features
        - Splitting into train/test sets
        """
        data = pd.read_csv(self.config.data_path)
        logger.info(f"Original data shape: {data.shape}")

        # Drop id column as it is statistically insignificant
        data = data.drop(labels=['id'], axis=1)
        logger.info("Dropped 'id' column")

        # Identify numerical and categorical columns
        numerical_columns = list(data.columns[data.dtypes != 'object'])
        categorical_columns = list(data.columns[data.dtypes == 'object'])

        logger.info(f"Numerical columns: {numerical_columns}")
        logger.info(f"Categorical columns: {categorical_columns}")

        # Handle missing values
        data = self._handle_missing_values(data, numerical_columns, categorical_columns)

        # Encode categorical variables
        data, label_encoders = self._encode_categorical(data, categorical_columns)

        # Separate feature columns from target column for scaling
        feature_numerical_columns = [col for col in numerical_columns if col != 'price']
        
        # Scale only feature numerical columns (exclude price)
        data, scaler = self._scale_numerical(data, feature_numerical_columns)

        # Split the data into training and test sets (0.80, 0.20 split)
        train, test = train_test_split(data, test_size=0.20, random_state=42)

        # Save preprocessed data
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        # Save preprocessing objects for prediction
        import joblib
        
        joblib.dump(label_encoders, os.path.join(self.config.root_dir, "label_encoders.joblib"))
        joblib.dump(scaler, os.path.join(self.config.root_dir, "scaler.joblib"))
        joblib.dump(feature_numerical_columns, os.path.join(self.config.root_dir, "numerical_columns.joblib"))
        joblib.dump(categorical_columns, os.path.join(self.config.root_dir, "categorical_columns.joblib"))
        logger.info("Saved preprocessing objects (encoders, scaler)")

        logger.info("Preprocessed data split into training and test sets")
        logger.info(f"Training set shape: {train.shape}")
        logger.info(f"Test set shape: {test.shape}")

        print(f"Training set shape: {train.shape}")
        print(f"Test set shape: {test.shape}")

        return train, test


    def _handle_missing_values(self, data, numerical_columns, categorical_columns):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # For numerical columns, fill with median
        for col in numerical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
                logger.info(f"Filled missing values in {col} with median")
        
        # For categorical columns, fill with mode
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
                logger.info(f"Filled missing values in {col} with mode")
        
        return data


    def _encode_categorical(self, data, categorical_columns):
        """Encode categorical variables using LabelEncoder"""
        logger.info("Encoding categorical variables...")
        
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
            logger.info(f"Encoded {col}")
        
        return data, label_encoders


    def _scale_numerical(self, data, numerical_columns):
        """Scale numerical features using StandardScaler"""
        logger.info("Scaling numerical features...")
        
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        logger.info(f"Scaled {len(numerical_columns)} numerical columns")
        
        return data, scaler