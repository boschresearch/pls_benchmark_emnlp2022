# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from src.evaluations.utils import calc_metrics


def train_svm(X_train_tfidf, X_test_tfidf, X_dev_tfidf, y_train, y_test, y_dev):
    """Train a classifier using the provided TF-IDF feature vectors and make a prediction 
    for the test set using the threshold defined using development set.

    Args:
        X_train_tfidf (numpy array): TF-IDF feature vector
        X_test_tfidf (numpy array): TF-IDF feature vector
        X_dev_tfidf (numpy array): TF-IDF feature vector 
        y_train (numpya array): target variable
        y_test (numpya array): target variable
        y_dev (numpya array): target variable

    Returns:
        dict: evaluation metrics
    """

    # Initialize a classifier.
    clf = OneVsRestClassifier(SVC()).fit(X_train_tfidf, y_train)

    # Find the decision function for best macro score
    metrics_dev = list()
    thresholds = []
    threshold = -1.0

    # Intiailize different thresholds for which you wish to evaluate the model.
    for index in range(1, 20):
        thresholds.append(threshold)
        threshold += 0.1
    
    # Search through different decision function thresholds.
    for threshold in thresholds:
        y_dev_pred = clf.decision_function(X_dev_tfidf)
        y_dev_pred = y_dev_pred > threshold
        perf = calc_metrics(y_dev, y_dev_pred, 'svc+tf-idf')
        perf["threshold"] = threshold
        metrics_dev.append(perf)

    df_metrics_dev = pd.DataFrame(metrics_dev)
    
    # Identify the best threshold for the dev split
    threshold = df_metrics_dev.threshold.iloc[df_metrics_dev.f1_macro.idxmax()]

    pred = clf.decision_function(X_test_tfidf)

    # Use the threshold to make a prediction
    pred = pred > threshold

    # Evaluate the model output
    return calc_metrics(y_test, pred, 'svc+tf-idf')
