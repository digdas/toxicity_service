from transformers import pipeline

def load_model():
    return pipeline("text-classification", model="cointegrated/rubert-tiny-toxicity", tokenizer="cointegrated/rubert-tiny-toxicity")

def predict_toxicity(model, text):
    result = model(text)
    labels = [res['label'] for res in result if res['score'] > 0.5]
    return labels
