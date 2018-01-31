from util import *
import os

os.environ["JOBLIB_START_METHOD"] = 'forkserver'

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG,
		    filename="./log/rf_training.log", filemode='w',datefmt='%m-%d %H:%M')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)-12s: %(levelname)-8s %(message)s',"%Y-%m-%d %H:%M:%S")
console.setFormatter(formatter)

logging.getLogger('').addHandler(console)


# Read data in
logging.info("Loading data..........")
X_train = feather.read_dataframe("./data/df_x_train.feather")
Y_train = feather.read_dataframe("./data/df_y_train.feather")

x_val = feather.read_dataframe("./data/X_val.feather")
y_val = feather.read_dataframe("./data/y_val.feather")

x_test = feather.read_dataframe("./data/X_test.feather")

weights = np.load("./data/weights.npy")


# Data normalization
logging.info("Performing data standardization.........")
train_scaler = StandardScaler()
train_scaler.fit(X_train)
X_train = train_scaler.transform(X_train)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
Y_train = Y_train.as_matrix()


val_scaler = StandardScaler()
val_scaler.fit(x_val)
test_scaler = StandardScaler()
test_scaler.fit(x_test)

x_val = val_scaler.transform(x_val)
x_val = x_val.reshape((x_val.shape[0],1,x_val.shape[1]))
x_test = val_scaler.transform(x_test)
x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
y_val = y_val.as_matrix()



# Model

logging.info("Building regressor.........")
regr = RandomForestRegressor(n_jobs=-1)
parameters = {'n_estimators':[50,80], 'min_samples_split':[10,15],
              'max_depth':[10,15]}

scoring = make_scorer(mean_squared_error)

grid_search_obj = GridSearchCV(estimator=regr, param_grid=parameters, 
                               scoring=scoring, n_jobs=-1,verbose=2)

grid_search_obj.fit(X_train[:170810*2,0,:], Y_train[:170810*2]-Y_train.mean())

best_reg = grid_search_obj.best_estimator_

rf_val_pred = best_reg.predict(x_val[:,0,:])+Y_train.mean()
rf_test_pred = best_reg.predict(x_test[:, 0, :])+Y_train.mean()

logging.info("Results for validation from Random forest......................................")
logging.info("The NWRMSLE is {0}".format(NWRMSLE(y_val, rf_val_pred, weights[:170810], if_list=False)))
logging.info("The r2 score is {0}".format(r2_score(y_val, rf_val_pred, sample_weight=weights[:170810])))
logging.info("The explained variance is {0}".format(explained_variance_score(y_val,rf_val_pred,sample_weight=weights[:170810])))
logging.info("Saving predictions...................")

np.save("./res/rf_grid_val.npy", rf_val_pred)
np.save("./res/rf_grid_test.npy", rf_test_pred)

logging.info("\n") 


def run_RF_one_day(best_reg, X, Y, w, xval, yval, xtest):
    pred_val = []
    pred_test = []
    
    parameters = best_reg.get_params()
    n = parameters['n_estimators']
    min_samples_split = parameters['min_samples_split']
    max_depth = parameters['max_depth']
    
    for i in range(0, 16):
        logging.info("Predicting %d day "%i)
        logging.info("="*50)
        
        y = Y[:,i] - Y[:,i].mean()
        reg = RandomForestRegressor(n_estimators=n, min_samples_split=min_samples_split,
                                   max_depth=max_depth,n_jobs=-1, verbose=2)
        reg.fit(X, y, sample_weight=w)
        
        pred_val.append(reg.predict(xval)+Y[:,i].mean())
        pred_test.append(reg.predict(xtest)+Y[:,i].mean())
        
    return pred_val, pred_test

logging.info("Training one day RF model.............................................")
one_day_pred_val, one_day_test_val = run_RF_one_day(best_reg, X_train[:,0,:], Y_train, weights,
                                    x_val[:,0,:], y_val, x_test[:,0,:])
logging.info("The NWRMSLE of validation set is {0}".format(NWRMSLE(y_val, np.array(one_day_pred_val).T, weights[:170810], if_list=False)))
logging.info("The r2 score of validation set with RF is {0}".format(r2_score(y_val, np.array(one_day_pred_val).T, sample_weight=weights[:170810])))
logging.info("The explained variance is {0}".format(explained_variance_score(y_val,np.array(one_day_pred_val).T,sample_weight=weights[:170810])))
logging.info("\n")

np.save('./res/one_day_pred_val.npy', one_day_pred_val)
np.save('./res/one_day_pred_test.npy', one_day_test_val)


def RF_feature_selection(best_reg, X, Y, w, xval, yval, xtest):
    pred_val = []
    pred_test = []
    
    parameters = best_reg.get_params()
    n = parameters['n_estimators']
    min_samples_split = parameters['min_samples_split']
    max_depth = parameters['max_depth']
    
    for i in range(0, 16):
        logging.info("Predicting %d day "%i)
        logging.info("="*50)
        
        y = Y[:,i] - Y[:,i].mean()
        reg = RandomForestRegressor(n_estimators=n, min_samples_split=min_samples_split,
                                   max_depth=max_depth,n_jobs=-1, verbose=2)
        
        reg_filter = SelectKBest(mutual_info_regression, k=300)
        
        new_reg = Pipeline([('filter', reg_filter),('reg', reg)])
        new_reg.fit(X, y, sample_weight=w)
        
        pred_val.append(reg.predict(xval)+Y[:,i].mean())
        pred_test.append(reg.predict(xtest)+Y[:,i].mean())
        
    return pred_val, pred_test

logging.info("Training RF model with feature selection.............................................")
fea_pred_val, fea_pred_test = run_RF_one_day(best_reg, X_train[:,0,:], Y_train[:,:], weights,
                                    x_val[:,0,:], y_val, x_test[:,0,:])
logging.info("The NWRMSLE of validation set is {0}".format(NWRMSLE(y_val, np.array(fea_pred_val).T, weights[:170810], if_list=False)))
logging.info("The r2 score of validation set with RF is {0}".format(r2_score(y_val, np.array(fea_pred_val).T, sample_weight=weights[:170810])))
logging.info("The explained variance is {0}".format(explained_variance_score(y_val,np.array(fea_pred_val).T,sample_weight=weights[:170810])))
logging.info("\n")

np.save('./res/fea_pred_val.npy',fea_pred_val)
np.save('./res/fea_pred_test.npy',fea_pred_test)
logging.info("Everything is Done")
