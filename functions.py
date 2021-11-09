import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor

import matplotlib.pyplot as plt

from scipy.stats import zscore

def load_data(filename):
    data=pd.read_csv(filename)
    df=pd.DataFrame(data,columns=['id','artist_id','artist','name','popularity','release_date','duration_ms',
                                  'explicit','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness',
                                  'liveness','valence','tempo','artist_genres','artist_followers','url',
                                  'playlist_id','playlist_description','playlist_name','playlist_url','query'])
    return df


def remove_outliers(dataframe):
    hdrs=dataframe.columns
    dataframe=np.array(dataframe,dtype=np.float64)
    ndf=dataframe[(np.abs(zscore(dataframe))<3).all(axis=1)]

    ndf=pd.DataFrame(ndf,columns=hdrs)
    return ndf


def grisearch(X_train, y_train,y_test,X_test):
    '''grid search'''
    tuned_parameters = [
        {"kernel": ["rbf"], "gamma": [0.1, 0.01, 0.001, 0.0001, .05, 0.2, 0.5], "C": [1, 10, 100, 1000]},
    ]

    print("# Tuning hyper-parameters")
    print()
    clf = GridSearchCV(svm.SVR(), tuned_parameters, scoring='r2')
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Best score set found on development set:")
    print()
    print(clf.best_score_)
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

def bagging_reg(X,y):
    model=BaggingRegressor()
    model.fit(X,y)
    prediction=model.predict(X)
    print('Bagging regression')
    print('R2: %.2f' % r2_score(y, prediction))
    print('MSE: %.2f' % mean_squared_error(y, prediction))



def boosting_reg(X,y):
    model=AdaBoostRegressor()
    model.fit(X,y)
    prediction=model.predict(X)
    print('Boosting regression')
    print('R2: %.2f' % r2_score(y, prediction))
    print('MSE: %.2f' % mean_squared_error(y, prediction))

def rezidual_graph(y_true,predict):
    plt.scatter(y_true,predict,c=['red','green'])
    plt.legend(['Hodnoty testovacej mnoziny','Predpoved'])
    plt.show()
