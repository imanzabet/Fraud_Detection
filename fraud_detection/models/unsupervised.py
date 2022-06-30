# importing libraries
from fraud_detection.models import regression as reg
from fraud_detection.main import preprocessing as pp
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Visualization Modules
import seaborn as sns
import matplotlib.pyplot as plt

# Feature Engineering Modules
from sklearn.preprocessing import PolynomialFeatures

# Metrics Modules
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, classification_report, confusion_matrix,
                             average_precision_score, precision_recall_curve,
                             roc_curve, roc_auc_score, roc_auc_score)

# Generalized Models Modules
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.cluster import KMeans

class Unsupervised(KMeans):
    def __init__(self):
        pass


class Clustering():
    def __init__(self):
        pass
    def kmeans(self, df, n_clusters, random_state):
        kmeans = KMeans(n_clusters=n_clusters, verbose=1, random_state=random_state)
        return kmeans.fit(df)
    def elbow_viz(self):
        # sns.lineplot(x=clustering_scores['clusters'], y=clustering_scores['scores'])
        plt.xlim(0.5, 10)


if False:
    df = pp().pickle_load(fname='dfPickle_2')
    df = df.dropna()
    columns = df.columns.tolist()
    # Filter the columns to remove data we do not want
    columns = [c for c in columns if c not in ['isFraud', 'index']]
    # Store the variable we are predicting
    target = 'isFraud'
    # Define a random state
    random_state = np.random.RandomState(22)



    x_train, x_test, y_train, y_test = train_test_split(df[columns], df[[target]],
                                                     test_size = 0.3, random_state = random_state)

    Fraud_len = len(y_train[y_train['isFraud'] == 1])
    Valid_len = len(y_train[y_train['isFraud'] == 0])
    outlier_fraction = Fraud_len/float(Valid_len)

    y_tr = y_train['isFraud'].tolist()
    y_ts = y_test['isFraud'].tolist()


    # Define the outlier detection methods

    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    classifiers = {
        "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(x_train),
                                           contamination=0.5,random_state=random_state, verbose=0),
        "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto',
                                                  leaf_size=30, metric='minkowski',
                                                  p=2, metric_params=None, contamination=0.5),
        "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05,
                                             max_iter=-1)
    }

    #Fit the model
    n_outliers = Fraud_len
    for i, (clf_name,clf) in enumerate(classifiers.items()):
        #Fit the data and tag outliers
        if clf_name == "Local Outlier Factor":
            y_pred = clf.fit_predict(x_train)
            scores_prediction = clf.negative_outlier_factor_
        elif clf_name == "Support Vector Machine":
            clf.fit(x_train)
            y_pred = clf.predict(x_train)
        else:
            clf.fit(x_train)
            scores_prediction = clf.decision_function(x_train)
            y_pred = clf.predict(x_train)
        #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        n_errors = (y_pred != y_tr).sum()
        # Run Classification Metrics
        print("{}: {}".format(clf_name,n_errors))
    #     print("Accuracy Score :")
    #     print(accuracy_score(y_tr,y_pred))
        print("Confustion Matrix :")
        bin_y_proba = [0 if x < 0.5 else 1 for x in y_pred]
        print(confusion_matrix(y_test, bin_y_proba))
        print("Classification Report :")
        print(classification_report(y_tr,y_pred))
    
if (__name__ == '__main__'):
    from fraud_detection.main import preprocessing
    pp = preprocessing()
    df = pp.pickle_load(fname='dfPickle_2')
    cluster = Clustering()
    kmeans_clusters = cluster.kmeans(df, 3, 22)