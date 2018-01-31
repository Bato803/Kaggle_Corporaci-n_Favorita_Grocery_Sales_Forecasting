import numpy as np
import pandas as pd
import feather 
import time
import matplotlib.pyplot as plt
import sys
import logging
default_stdout = sys.stdout
import pdb

# Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, GRU
from keras.layers.advanced_activations import ELU, PReLU
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger


# Sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error,r2_score,explained_variance_score
from sklearn.feature_selection import mutual_info_regression,SelectKBest
from sklearn.pipeline import Pipeline


def NWRMSLE(y_true, y_pred, weights, if_list=True):
    """
    y_pred: a list of prediction of length 16. 
    """
    if if_list:
        error = (y_true - np.array(y_pred).squeeze(axis=2).transpose())**2
    else:
        error = (y_true-y_pred)**2
    
    normalized_weighted_error = error.sum(axis=1)*weights
    root_mean = np.sqrt(normalized_weighted_error.sum()/weights.sum()/16)
    
    return root_mean


def create_submission(df_2017, df_test):
    y_test = np.load("./res/lstm_pred_test_5.npy")
    y_test = y_test.squeeze(axis=2).transpose()
    df_preds = pd.DataFrame(y_test, index=df_2017.index,
                           columns=pd.date_range("2017-08-16", periods=16)).stack().to_frame("unit_sales")

    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
    
    submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
    submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
    submission.to_csv('./submission/lstm_pred_test_5.csv', float_format='%.4f', index=None)
    
    return











