from transformers import pipeline
from app.logger import logger  # Import the logger

def load_model():
    return pipeline("text-classification", model="cointegrated/rubert-tiny-toxicity", tokenizer="cointegrated/rubert-tiny-toxicity", return_all_scores=True)

def predict_toxicity(model, text):
    # Predict the toxicity for the input text
    predictions = model(text)  # Assuming you have a method to predict toxicity

    # Log the input text and prediction result
    logger.info(f"Input text: {text}")
    logger.info(f"Predictions: {predictions}")

    # Map the prediction results to labels and scores
    labels_with_scores = []
    for predict in predictions[0]:  # Assuming predictions return a label-score tuple
        labels_with_scores.append({"label": predict['label'], "score": predict['score']})

    return labels_with_scores