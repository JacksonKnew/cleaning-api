import logging
from fastapi import FastAPI
import control.control as ctrl
import utils.request_classes as rq

logging.basicConfig(level=logging.INFO)
app = FastAPI()
controller = ctrl.Controller()


@app.get("/pipeline")
def get_pipeline():
    """GET at /pipeline
    This endpoint serves to get the pipeline used for segmenting emails

    Returns:
    A json containing the pipeline structure
    """
    return controller.get_pipeline_to_json()


@app.put("/pipeline")
def set_pipeline(pipeline_name: rq.PipelineName):
    """PUT at /pipeline
    This endpoint serves to change the pipeline used for segmenting emails

    Expected arguments:
    - name: name of the pipeline to use from now on

    Returns:
    A json containing the pipeline structure
    """
    controller.set_pipeline_from_json(pipeline_name)
    return controller.get_pipeline_to_json()


@app.post("/segment")
def segmentation(email: rq.ThreadList):
    """POST at /segment
    This endpoint serves to segment a list of emails

    Expected arguments:
    - threads: list of emails in the form of single character strings

    Returns:
    - threads: list of
        - thread:
            - source: source text of the email, untreated
            - messages:
                - message:
                    - header
                    - disclaimer
                    - greetings
                    - body
                    - signature
                    - caution
    """
    controller.set_dataset_from_json(email)
    controller.segment()
    return controller.get_dataset_to_json()


@app.put("/train_classifier")
def train_classifier(train_request: rq.TrainRequest):
    """POST at /train
    This endpoint serves to train the classifier of the current model

    Expected arguments:
    - csv_path: path to csv file containing the training data
    - epochs: number of epochs to train for
    """
    controller.set_dataset_from_csv(train_request.csv_path)
    controller.train_classifier(train_request.epochs)
    return controller.get_pipeline_to_json()


@app.put("/train_encoder")
def train_encoder(train_request: rq.TrainRequest):
    """POST at /train
    This endpoint serves to train the encoder of the current model

    Expected arguments:
    - csv_path: path to csv file containing the training data
    - epochs: number of epochs to train for
    """
    controller.set_line_dataset_from_csv(train_request.csv_path)
    controller.train_encoder(train_request.epochs)
    return controller.get_pipeline_to_json()
