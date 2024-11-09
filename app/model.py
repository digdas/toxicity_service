from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from app.logger import logger

MAX_LENGTH = 512  # Maximum token length for the model
MODEL_PATH = "/app/app/toxicity_model"

# Load model and tokenizer
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

# Function to split text into chunks of 512 tokens
def split_text(text, tokenizer, max_length=MAX_LENGTH):
    # Tokenize the text and get input_ids
    tokens = tokenizer.encode(text, truncation=False)
    
    # Split into chunks of max_length tokens
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    
    # Decode the chunks back to text
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Function to predict toxicity and aggregate results
def predict_toxicity(model, tokenizer, text):
    # Convert text to lower case for better predictions
    text = text.lower()

    # Split text into chunks if it's too long (over 512 tokens)
    if len(tokenizer.encode(text)) > MAX_LENGTH:
        text_chunks = split_text(text, tokenizer)
    else:
        text_chunks = [text]

    # Dictionary to store aggregated scores
    aggregated_scores = {label: 0 for label in ["NORMAL", "INSULT", "THREAT", "OBSCENITY", "PROFANITY"]}

    # Loop through each chunk and make predictions
    for chunk in text_chunks:
        # Tokenize the chunk
        inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {key: val.to(model.device) for key, val in inputs.items()}
        
        # Get predictions (use torch.no_grad() to avoid gradient computation)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        # Aggregate the scores for each label
        for label, prob in zip(aggregated_scores.keys(), probabilities):
            aggregated_scores[label] += prob

    # Average the scores across chunks
    num_chunks = len(text_chunks)
    aggregated_scores = {label: score / num_chunks for label, score in aggregated_scores.items()}

    # Return aggregated scores as a list of dictionaries
    labels_with_scores = [{"label": label, "score": score} for label, score in aggregated_scores.items()]

    logger.info(f"labels_with_scores: {labels_with_scores}")

    return labels_with_scores