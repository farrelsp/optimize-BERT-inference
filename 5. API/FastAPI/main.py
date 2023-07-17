import time
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

import torch
from torch import optim
import torch.nn.functional as F

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
from onnxruntime import InferenceSession

app = FastAPI(
    title="Sentiment Model API",
    version="0.1",
)

path = "./../../model/indobert-onnx"
model_path = path + "/quantized_optimized.onnx"
session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
tokenizer = BertTokenizer.from_pretrained(path)

i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}

class Item(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/sentiment/predict/")
async def predict(review: Item):
    start = time.time()

    # Tokenize input
    inputs = tokenizer([review.text])
    inputs_onnx = dict(
        input_ids=np.array(inputs["input_ids"]).astype("int64"),
        attention_mask=np.array(inputs["attention_mask"]).astype("int64"),
        token_type_ids=np.array(inputs["token_type_ids"]).astype("int64")
    )

    # Model Inference
    logits = session.run(None, input_feed=inputs_onnx)[0]
    label = torch.topk(torch.from_numpy(logits), k=1, dim=-1)[1].squeeze().item()
    probability = F.softmax(torch.from_numpy(logits), dim=-1).squeeze()[label].item()

    end = time.time()
    duration = end - start

    result = {
        "text": review.text,
        "label": i2w[label],
        "probability": probability,
        "duration": duration
    }

    return result