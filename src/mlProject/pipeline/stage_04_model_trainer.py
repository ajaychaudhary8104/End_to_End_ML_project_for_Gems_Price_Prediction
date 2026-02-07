import sys
import os
from pathlib import Path

# Add parent directories to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_trainer import ModelTrainer
from src.mlProject import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()


if __name__ == "__main__":
    pipeline = ModelTrainerTrainingPipeline()
    pipeline.main()