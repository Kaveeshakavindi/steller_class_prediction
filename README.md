# Steller Class Prediction

This model was built to classify steller class as a submission for the the www.kaggle.com Predicting Stellar Class Playground Series - Season 6 Episode 6 competition. 

## Files
### preprocess.py
**one hot encoding** : convert each feature categorical column into numeric data

**outliers** : find outliers using seaborn boxplot

**Boxplot of u**
![Alt text](https://github.com/Kaveeshakavindi/steller_class_prediction/blob/main/images/u.png)

**Boxplot of g**
![Alt text](https://github.com/Kaveeshakavindi/steller_class_prediction/blob/main/images/g.png)

**Boxplot of r**
![Alt text](https://github.com/Kaveeshakavindi/steller_class_prediction/blob/main/images/r.png)

**Boxplot of i**
![Alt text](https://github.com/Kaveeshakavindi/steller_class_prediction/blob/main/images/i.png)

**Boxplot of z**
![Alt text](https://github.com/Kaveeshakavindi/steller_class_prediction/blob/main/images/z.png)

**Boxplot of redshift**
![Alt text](https://github.com/Kaveeshakavindi/steller_class_prediction/blob/main/images/redshift.png)

**robust_scaler_**: reduce impact of outliers using robust scaler

### model.py
The model based on Random Forest Classifier achived 96% accuracy. 

![Alt text](https://github.com/Kaveeshakavindi/steller_class_prediction/blob/main/images/conf.png)


