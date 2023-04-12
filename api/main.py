import os
import logging

logging.basicConfig(level=logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from fastapi import FastAPI
import control.control as ctrl
import model.request_classes as rq

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


@app.post("/train")
def train(train_request: rq.TrainRequest):
    """POST at /train
    This endpoint serves to train a model

    Expected arguments:
    - model_name: name of the model to train
    - metrics: list of metrics to use for training
    - loss_weights: list of weights to use for each metric
    - epochs: number of epochs to train for

    Returns:
    - model_name: name of the trained model
    - metrics: final list of metrics for trained model
    - loss: final loss for trained model
    - epochs: number of epochs trained for
    """
    pass
