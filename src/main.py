import copy

import sklearn
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import logging
import os
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats import mannwhitneyu, sem
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFE, RFECV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.datasets import make_classification

random_state = 42
classifiers = [
    RidgeClassifierCV(),
    RandomForestClassifier(),
    IsolationForest(),
    SVC(),
    DecisionTreeClassifier(),
    BaggingClassifier(),
    LogisticRegressionCV(),
    ExtraTreeClassifier(),
    SGDClassifier(),
    PassiveAggressiveClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    SGDClassifier(), RidgeClassifier(), RidgeClassifierCV(), PassiveAggressiveClassifier(),
    AdaBoostClassifier(), GradientBoostingClassifier(),
    BaggingClassifier(), ExtraTreesClassifier(),
    LogisticRegression(), LogisticRegressionCV(),
    KNeighborsClassifier(),
    GaussianProcessClassifier(),
    BernoulliNB(), GaussianNB(), LinearDiscriminantAnalysis(), LinearSVC(),
    MultinomialNB(), NearestCentroid(),
    NuSVC(), QuadraticDiscriminantAnalysis(),
]

n_splits = 10
n_repeats = 10
log_file = "./log/logging.log"

# initialize logging
log_dir = os.path.dirname(log_file)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(filename=log_file,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# load dataset
X, y = load_wine(return_X_y=True)
logging.info('Shape: %s, %s' % (X.shape, y.shape))
# minimally prepare dataset
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))

# evaluate naive
naive = DummyClassifier(strategy='most_frequent')

scores = pd.DataFrame()
fake_scores = pd.DataFrame()

for classifier in classifiers:

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    if hasattr(classifier, "random_state"):
        classifier.random_state = random_state

    pipeline = Pipeline([("scaler", MinMaxScaler()), ("classifier", copy.deepcopy(classifier))])

    true_scores = cross_validate(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1,
                                 error_score='raise', return_train_score=True)

    # fake dataset, that should have:
    # - same shape for X
    # - same distribution of labels for y, only scrambled

    # fix random seed
    logging.info("Randomizing dataset...")
    np.random.seed(42)
    X_fake = np.random.rand(X.shape[0], X.shape[1])
    np.random.shuffle(y)

    fake_scores = cross_validate(pipeline, X_fake, y, scoring='accuracy', cv=cv, n_jobs=-1,
                                 error_score='raise', return_train_score=True)

    score_summary = {}
    score_summary["classifier"] = classifier.__class__.__name__
    score_summary["train_score_avg"] = mean(true_scores["train_score"])
    score_summary["train_score_sem"] = sem(true_scores["train_score"])
    score_summary["test_score_avg"] = mean(true_scores["test_score"])
    score_summary["test_score_sem"] = sem(true_scores["test_score"])
    score_summary["fake_train_score_avg"] = mean(fake_scores["train_score"])
    score_summary["fake_train_score_sem"] = sem(fake_scores["train_score"])
    score_summary["fake_test_score_avg"] = mean(fake_scores["test_score"])
    score_summary["fake_test_score_sem"] = sem(fake_scores["test_score"])
    scores = pd.concat([scores, pd.Series(score_summary).to_frame()], ignore_index=True, axis=1)

results_dir = "./results"
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
scores.T.to_csv(os.path.join(results_dir, "scores.csv"))
