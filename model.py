from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def random_forest_classifier_model(X_train, y_train, X_test, y_test, train_data_cols_y):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # save model
    joblib.dump(classifier, 'random_forest_model.pkl')

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'accuracy: {accuracy*100:.2f}%') 
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=train_data_cols_y, yticklabels=train_data_cols_y)

    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted labels')
    plt.ylabel('True Labels')
    plt.show()