import torch
import evaluate
import numpy as np
import gradio as gr

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

dataset = load_dataset("dar-ai/emotion", "split")

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def tokenize(text):
    tokens = tokenizer(text["text"], padding=True, truncation=True)
    return tokens

train_set = dataset["train"].map(tokenize, batched=True)
valid_set = dataset["validation"].map(tokenize, batched=True)

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

training_args = TrainingArguments(
    output_dir="sentiment-checkpoint",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    model_accuracy = accuracy_metric.compute(references=labels, predictions=predictions)
    model_precision = precision_metric.compute(references=labels, predictions=predictions, average="macro")
    model_recall = recall_metric.compute(references=labels, predictions=predictions, average="macro")
    evaluation = {"eval_accuracy": model_accuracy, "eval_precision":model_precision, "eval_recall":model_recall}
    return evaluation

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model("sentiment-model")

analyzer = pipeline("sentiment-analysis", model="sentiment-model")
options = {0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"}
pred = analyzer("Today is a beautiful day")[0]
label = options[int(pred["label"][-1])]
score = pred["score"]*100
print(f"Sentiment: {label}   Score: {score:.4f}%")

def predict(text):
    pred = analyzer(text)[0]
    label = options[int(pred["label"[-1]])]
    return label

options = {0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"}
interface = gr.Interface(fn=predict, inputs="text", outputs="text")
interface.launch(share=True)
