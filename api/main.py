import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from fastapi import FastAPI
import model.data as data
import model.pipelining as pipe
import model.request_classes as rq

app = FastAPI()

pipeline = pipe.PipelineModel("default")


@app.get("/pipeline")
def get_pipeline():
    """GET at /pipeline
    This endpoint serves to get the pipeline used for segmenting emails

    Returns:
    A json containing the pipeline structure
    """
    return pipeline.to_json()


@app.put("/pipeline")
def set_pipeline(pipeline_name: rq.PipelineName):
    """PUT at /pipeline
    This endpoint serves to change the pipeline used for segmenting emails

    Expected arguments:
    - name: name of the pipeline to use from now on

    Returns:
    A json containing the pipeline structure
    """
    pipeline.from_json(pipeline_name)
    return pipeline.to_json()


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
    dataset = data.EmailDataset.from_json(email)
    dataset.segment(pipeline)
    return dataset.to_json()
