from transformers import pipeline
from app.logger import logger

MAX_LENGTH = 512  # Maximum token length for the model

def load_model():
    return pipeline("text-classification", model="cointegrated/rubert-tiny-toxicity", tokenizer="cointegrated/rubert-tiny-toxicity", return_all_scores=True)

def split_text(text, tokenizer, max_length):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def predict_toxicity(model, text):
    # Convert text to lower case for better predictions
    text = text.lower()
    # Split text into chunks if it's too long
    tokenizer = model.tokenizer
    if len(tokenizer.encode(text)) > MAX_LENGTH:
        text_chunks = split_text(text, tokenizer, MAX_LENGTH)
        logger.info("Text split into chunks due to length.")
    else:
        text_chunks = [text]

    aggregated_scores = {}

    for chunk in text_chunks:
        predictions = model(chunk)

        # Log each chunk and prediction
        logger.info(f"Chunk: {chunk}")
        logger.info(f"Predictions: {predictions}")

        # Aggregate scores for each label across all chunks
        for prediction in predictions[0]:  # Assuming predictions return a list of label-score dictionaries
            label = prediction['label']
            score = prediction['score']
            if label in aggregated_scores:
                aggregated_scores[label] += score
            else:
                aggregated_scores[label] = score

    # Average the scores by dividing by the number of chunks
    num_chunks = len(text_chunks)
    aggregated_scores = {label: score / num_chunks for label, score in aggregated_scores.items()}

    # Map the aggregated scores to labels and return as a list of dicts
    labels_with_scores = [{"label": label, "score": score} for label, score in aggregated_scores.items()]
    return labels_with_scores
