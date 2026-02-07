import sys
import os
from pathlib import Path

# Add parent directories to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_evaluation import ModelEvaluation
from src.mlProject import logger


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow()


if __name__ == "__main__":
    pipeline = ModelEvaluationPipeline()
    pipeline.main()
