import pandas as pd
from sklearn import preprocessing
import joblib

# one hot encode
def one_hot_encoding(feature, train_data):
    df = pd.get_dummies(train_data[feature]).astype(int)
    train_data = pd.concat([train_data, df], axis=1).reindex(train_data.index)
    train_data.drop(feature, axis=1, inplace=True)
    return train_data

# scaling
def robust_scalar_train(X_train, cols_to_scale, cols_binary):
    scaler = preprocessing.RobustScaler()
    df = scaler.fit_transform(X_train[cols_to_scale])
    df = pd.DataFrame(df, columns=cols_to_scale)
    x_train_scaled = pd.concat([df, X_train[cols_binary].reset_index(drop=True)], axis=1)
    joblib.dump(scaler, 'robust_scaler.pkl') 
    return x_train_scaled, scaler

def robust_scalar_test(X_test, scaler, cols_to_scale, cols_binary):
    df = scaler.transform(X_test[cols_to_scale])
    df = pd.DataFrame(df, columns=cols_to_scale)
    x_test_scaled = pd.concat([df, X_test[cols_binary].reset_index(drop=True)], axis=1)
    return x_test_scaled
