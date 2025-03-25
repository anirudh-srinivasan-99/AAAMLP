from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold


class GreedyFeatureSelection:
    def __init__(self):
        pass

    def evaluate_score(self, x: np.ndarray, y: np.ndarray) -> float:
        model = linear_model.LogisticRegression()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict_proba(x_test)[:, 1]
            auc_scores.append(metrics.roc_auc_score(y_test, y_pred))


        return np.mean(auc_scores)
    
    def _feature_selection(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[List[str], List[float]]:
        best_features = []
        best_scores = []

        num_features = X.shape[1]

        while True:
            selected_feature = None
            best_score = 0

            for feature in range(num_features):
                if feature in best_features:
                    continue

                selected_features = [feature] + best_features
                score = self.evaluate_score(X[:, selected_features], y)

                if score > best_score:
                    selected_feature = feature
                    best_score = score
            
            if selected_feature != None:
                best_features.append(selected_feature)
                best_scores.append(best_score)
            
            if len(best_scores) > 2 and best_scores[-1] < best_scores[-2]:
                break
        
        # Removing the last entry as it led to a reduction in Accuracy.
        return best_features[:-1], best_scores[:-1]

    def __call__(self, X, y) -> Tuple[np.ndarray, float]:
        best_features, scores = self._feature_selection(X, y)
        return X[:, best_features], scores

if __name__ == '__main__':
    x, y = make_classification(n_samples=1000, n_classes=2, n_features=100)

    x_transformed, scores = GreedyFeatureSelection()(x, y)

    print(f'X before feature selection {x.shape}')
    print(f'X after feature selection: {x_transformed.shape}, score: {scores[-1]}')



