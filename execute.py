import pandas as pd
from preprocess import one_hot_encoding, robust_scalar_test, robust_scalar_train
from sklearn.model_selection import train_test_split
from model import random_forest_classifier_model

# load data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

# dropna
train_data = train_data.dropna()

# one hot encoding
train_data = one_hot_encoding('galaxy_population', train_data)
train_data = one_hot_encoding('spectral_type', train_data)

train_data_cols_x = ['id', 'alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 'Blue_Cloud', 'Red_Sequence', 'A/F', 'G/K', 'M', 'O/B']
train_data_cols_y = ['class']

X_train = train_data[train_data_cols_x]
X_test = train_data[train_data_cols_y]

# split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, X_test, test_size=0.2, random_state=42)

# scale
cols_to_scale = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
cols_binary   = ['Blue_Cloud', 'Red_Sequence', 'A/F', 'G/K', 'M', 'O/B', 'id']
X_train_split_scaled, scalar = robust_scalar_train(X_train_split, cols_to_scale, cols_binary)
X_test_split_scaled = robust_scalar_test(X_test_split, scalar, cols_to_scale, cols_binary)

# model
random_forest_classifier_model(X_train_split_scaled, y_train_split, X_test_split_scaled, y_test_split, train_data_cols_y)




