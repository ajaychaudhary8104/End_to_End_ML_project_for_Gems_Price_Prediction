## End_to_End_ML_project_for_Gems_Price_Prediction

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/ajaychaudhary8104/End_to_End_ML_project_for_Gems_Price_Prediction
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n gems python=3.12 -y
```

```bash
conda activate gems
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```






## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/ajaychaudhary8104/End_to_End_ML_project_for_Gems_Price_Prediction.mlflow \
MLFLOW_TRACKING_USERNAME=ajaychaudhary8104 \
MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/ajaychaudhary8104/End_to_End_ML_project_for_Gems_Price_Prediction.mlflow

export MLFLOW_TRACKING_USERNAME=ajaychaudhary8104

export MLFLOW_TRACKING_PASSWORD= your password

```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)