from transformers import pipeline # If using Hugging Face
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"): # Example Hugging Face model
        """
        Initializes the sentiment analyzer.
        Chooses between VADER, TextBlob, or Hugging Face.
        """
        self.model_name = model_name
        self.sentiment_pipeline = None

        # --- Option 1: Hugging Face (Recommended for advanced sentiment) ---
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model_name)
            print(f"Using Hugging Face sentiment model: {self.model_name}")
        except Exception as e:
            print(f"Could not load Hugging Face model {self.model_name}. Error: {e}")
            print("Falling back to VADER for sentiment analysis.")
            self._init_vader()

    def _init_vader(self):
        """Initializes VADER as a fallback."""
        self.analyzer = SentimentIntensityAnalyzer()
        self.sentiment_pipeline = None # Ensure HF pipeline is None

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of the given text.
        Returns a dictionary with sentiment information.
        Example: {'label': 'POSITIVE', 'score': 0.99} or {'compound': 0.8}
        """
        if self.sentiment_pipeline:
            result = self.sentiment_pipeline(text)[0]
            # Map common HF labels to a simpler sentiment type
            sentiment_label = "neutral"
            if "positive" in result['label'].lower() or result['label'] == 'LABEL_2':
                sentiment_label = "positive"
            elif "negative" in result['label'].lower() or result['label'] == 'LABEL_0':
                sentiment_label = "negative"
            return {"label": sentiment_label.upper(), "score": result['score']}
        elif hasattr(self, 'analyzer'): # VADER fallback
            vs = self.analyzer.polarity_scores(text)
            compound_score = vs['compound']
            if compound_score >= 0.05:
                return {"label": "POSITIVE", "score": compound_score}
            elif compound_score <= -0.05:
                return {"label": "NEGATIVE", "score": compound_score}
            else:
                return {"label": "NEUTRAL", "score": compound_score}
        else:
            return {"label": "UNKNOWN", "score": 0.0}

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print("Test 1 (Positive):", analyzer.analyze_sentiment("I love this product! It's fantastic."))
    print("Test 2 (Negative):", analyzer.analyze_sentiment("This is terrible, I'm very disappointed."))
    print("Test 3 (Neutral):", analyzer.analyze_sentiment("The weather is cloudy today."))
    print("Test 4 (Mixed):", analyzer.analyze_sentiment("The product is okay, but the service was bad."))