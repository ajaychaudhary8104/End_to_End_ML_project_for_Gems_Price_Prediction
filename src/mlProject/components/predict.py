import os
import joblib
import pandas as pd
from src.mlProject.entity.config_entity import PredictionConfig
from src.mlProject import logger


class Predictor:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = self._load_model()
        self.label_encoders = self._load_label_encoders()
        self.scaler = self._load_scaler()
        self.numerical_columns = self._load_numerical_columns()
        self.categorical_columns = self._load_categorical_columns()
        
    def _load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model not found at {self.config.model_path}")
        model = joblib.load(self.config.model_path)
        logger.info(f"Model loaded from {self.config.model_path}")
        return model
    
    def _load_label_encoders(self):
        """Load saved label encoders"""
        encoders_path = os.path.join(os.path.dirname(self.config.model_path), "..", "data_transformation", "label_encoders.joblib")
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
            logger.info(f"Label encoders loaded from {encoders_path}")
            return encoders
        logger.warning("Label encoders not found, using empty dict")
        return {}
    
    def _load_scaler(self):
        """Load saved scaler"""
        scaler_path = os.path.join(os.path.dirname(self.config.model_path), "..", "data_transformation", "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
            return scaler
        logger.warning("Scaler not found")
        return None
    
    def _load_numerical_columns(self):
        """Load numerical columns list"""
        cols_path = os.path.join(os.path.dirname(self.config.model_path), "..", "data_transformation", "numerical_columns.joblib")
        if os.path.exists(cols_path):
            cols = joblib.load(cols_path)
            logger.info(f"Numerical columns loaded: {cols}")
            return cols
        return ['carat', 'depth', 'table', 'x', 'y', 'z']
    
    def _load_categorical_columns(self):
        """Load categorical columns list"""
        cols_path = os.path.join(os.path.dirname(self.config.model_path), "..", "data_transformation", "categorical_columns.joblib")
        if os.path.exists(cols_path):
            cols = joblib.load(cols_path)
            logger.info(f"Categorical columns loaded: {cols}")
            return cols
        return ['cut', 'color', 'clarity']
    
    def preprocess_single_input(self, data: dict) -> pd.DataFrame:
        """
        Preprocess a single prediction input using the same logic as training
        
        Args:
            data: Dictionary with features {carat, cut, color, clarity, depth, table, x, y, z}
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Define feature columns (exclude id and price if present)
        feature_columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required feature: {col}")
        
        # Select only feature columns and ensure correct order
        df = df[feature_columns]
        
        logger.info(f"Input before preprocessing: {df.to_dict('records')[0]}")
        
        # Encode categorical variables using saved encoders
        for col in self.categorical_columns:
            if col in df.columns and col in self.label_encoders:
                try:
                    df[col] = self.label_encoders[col].transform(df[col])
                    logger.info(f"Encoded {col}: {df[col].values}")
                except ValueError as e:
                    # Handle unseen categories
                    logger.warning(f"Category not seen during training for {col}: {df[col].values[0]}")
                    raise ValueError(f"Invalid {col} value: {df[col].values[0]}")
        
        # Scale numerical features using saved scaler
        if self.scaler is not None:
            df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
            logger.info(f"Scaled numerical features")
        
        logger.info(f"Preprocessed input shape: {df.shape}")
        logger.info(f"Preprocessed input:\n{df}")
        
        return df
    
    def predict(self, data: dict) -> float:
        """
        Make prediction for a single input
        
        Args:
            data: Dictionary with gemstone features
            
        Returns:
            Predicted price
        """
        try:
            processed_data = self.preprocess_single_input(data)
            prediction = self.model.predict(processed_data)[0]
            logger.info(f"Prediction made: ${prediction:.2f}")
            return float(prediction)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
