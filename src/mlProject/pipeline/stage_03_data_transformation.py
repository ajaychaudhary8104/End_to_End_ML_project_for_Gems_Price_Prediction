import sys
import os
from pathlib import Path

# Add parent directories to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.data_transformation import DataTransformation
from src.mlProject import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train, test = data_transformation.preprocessing()


if __name__ == "__main__":
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()