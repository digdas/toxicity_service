from transformers import pipeline

def load_model():
    return pipeline("text-classification", model="cointegrated/rubert-tiny-toxicity", tokenizer="cointegrated/rubert-tiny-toxicity")

def predict_toxicity(model, text):
    # Predict the toxicity for the input text
    predictions = model(text)  # Assuming you have a method to predict toxicity

    # Map the prediction results to labels and scores
    labels_with_scores = []
    for label, score in predictions:  # Assuming predictions return a label-score tuple
        labels_with_scores.append({"label": label, "score": score})

    return labels_with_scores