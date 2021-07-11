import uvicorn
from scipy.sparse import issparse
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model, predict
from datetime import datetime, timezone

app = FastAPI(
    title="Iris Predictor",
    docs_url="/"
)

app.add_event_handler("startup", load_model)

class QueryIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class FeedbackIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    flower_class: str

class QueryOut(BaseModel):
    flower_class: str

@app.get("/ping")
def ping():
    return {"ping": "pong"}

#@app.post("/feedback_loop", status_code=200)
#def feedback_loop(
#        data: list[FeedbackIn]
#):
#    retrain(data)
#    return {"detail": "Feedback loop successful"}


@app.get("/Time")
async def root(start_date: datetime = datetime.now(timezone.utc)):
    print(start_date)
    return {"Time_Stamp": start_date}

@app.post("/predict_flower", response_model=QueryOut, status_code=200)
def predict_flower(
    query_data: QueryIn
  ):
    output = {'flower_class': predict(query_data)}
    return output


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8888, reload=True)
