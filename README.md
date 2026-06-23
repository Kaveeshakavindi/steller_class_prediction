# Steller Class Prediction

This model was built to classify steller class as a submission for the the www.kaggle.com Predicting Stellar Class Playground Series - Season 6 Episode 6 competition. 

## Files
### preprocess.py
**one hot encoding** : convert each feature categorical column into numeric data

**outliers** : find outliers using seaborn boxplot

**Boxplot of u**
![Alt text]('u.png')

**Boxplot of g**
![Alt text]('g.png')

**Boxplot of r**
![Alt text]('r.png')

**Boxplot of i**
![Alt text]('i.png')

**Boxplot of z**
![Alt text]('z.png')

**Boxplot of redshift**
![Alt text]('redshift.png')

**robust_scaler_**: reduce impact of outliers using robust scaler

### model.py
The model based on Random Forest Classifier achived 96% accuracy. 

![Alt text]('conf.png')


