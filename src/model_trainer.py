import logging
from typing import Dict, Any
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models: Dict[str, Any] = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }

    def train_models(self, X_train, y_train):
        """Trains multiple machine learning models."""
        logger.info("Training models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            logger.info(f"{name} trained.")

    def evaluate_models(self, X_test, y_test):
        """Evaluates trained models and prints reports."""
        logger.info("Evaluating models...")
        
        for name, model in self.models.items():
            print(f"\n{'='*20} {name} {'='*20}")
            predictions = model.predict(X_test)
            print(classification_report(y_test, predictions))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, predictions))
