import sys
import os
from pathlib import Path

# Add parent directories to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.predict import Predictor
from src.mlProject import logger


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, data: dict) -> float:
        """
        Make a prediction using trained model
        
        Args:
            data: Dictionary with gemstone features
            
        Returns:
            Predicted price
        """
        config = ConfigurationManager()
        prediction_config = config.get_prediction_config()
        predictor = Predictor(config=prediction_config)
        price = predictor.predict(data)
        return price


if __name__ == "__main__":
    # Example usage
    test_data = {
        'carat': 0.5,
        'cut': 'Ideal',
        'color': 'E',
        'clarity': 'SI1',
        'depth': 61.5,
        'table': 55.0,
        'x': 5.1,
        'y': 5.2,
        'z': 3.2
    }
    
    pipeline = PredictionPipeline()
    try:
        price = pipeline.predict(test_data)
        logger.info(f"Predicted price: ${price:.2f}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
