import joblib
import pandas as pd
from preprocess import robust_scalar_test
from preprocess import one_hot_encoding

# load model and scaler
classifier = joblib.load('random_forest_model.pkl')
scaler = joblib.load('robust_scaler.pkl')  

# load data
test_data = pd.read_csv('Data/test.csv')
submission_data = pd.read_csv('Data/sample_submission.csv')

# drop
test_data = test_data.dropna()

# one hot encoding
test_data = one_hot_encoding('galaxy_population', test_data)
test_data = one_hot_encoding('spectral_type', test_data)

# scale
cols_to_scale = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
cols_binary   = ['Blue_Cloud', 'Red_Sequence', 'A/F', 'G/K', 'M', 'O/B', 'id']
X_test_scaled = robust_scalar_test(test_data, scaler, cols_to_scale, cols_binary)


# predict
submission_ids = submission_data['id']
pred_for_submission_data = X_test_scaled[X_test_scaled['id'].isin(submission_ids)]
predictions = classifier.predict(pred_for_submission_data)
print(predictions)

output = pd.DataFrame({
    'id': submission_ids,
    'class': predictions
})
output.to_csv('submission.csv', index=False)
print(output.head())
    