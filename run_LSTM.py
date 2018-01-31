from util import * 
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG,
		    filename="./log/lstm_training.log", filemode='w',datefmt='%m-%d %H:%M')
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


def build_LSTM(X):
    
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1],X.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(128))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(64))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(32))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(16))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(1))
    
    return model


def run_LSTM(X, Y, xval, yval, xtest, weights, epoches):
    pred_val = []
    pred_test = []
    for i in range(0, 16):
        print("Predicting %d day "%i)
        print("="*50)
        
        y = Y[:,i] - Y.mean()
        model = build_LSTM(X)
        opt = optimizers.Adam(lr=0.001)
        model.compile(loss='mse', optimizer=opt, metrics=['mse'])
        
        call_backs = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1, 
                              verbose=1, mode='min'),
            CSVLogger('./log/lstm_training.csv', append=True, separator=';')
        ]
        
        model.fit(X, y, batch_size=4096, epochs=epoches, verbose=2,
                 sample_weight=weights, validation_data=(xval, yval[:,i]-Y.mean()),
                 callbacks=call_backs)
        
        pred_val.append(model.predict(xval)+Y.mean())
        pred_test.append(model.predict(xtest)+Y.mean())
        
    return model, pred_val, pred_test


logging.info("Training LSTM model.............................................")
model, pred_val, pred_test = run_LSTM(X_train, Y_train, x_val, y_val, 
                                     x_test, weights, 500)

np.save('./res/lstm_pred_val.npy', pred_val)
np.save('./res/lstm_pred_test.npy', pred_test)

logging.info("The NWRMSLE of validation set is {0}".format(NWRMSLE(y_val, pred_val, weights[:170810])))
logging.info("The r2 score of validation set with LSTM is {0}".format(r2_score(y_val, np.array(pred_val)[:,:,0].T, sample_weight=weights[:170810])))
logging.info("The explained variance is {0}".format(explained_variance_score(y_val,np.array(pred_val)[:,:,0].T,sample_weight=weights[:170810])))
logging.info("\n")
logging.info("LSTM is done")






