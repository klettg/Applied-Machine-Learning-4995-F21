#!/usr/bin/env python
# coding: utf-8

# # Homework 2: Decision Trees
# Due 10/21 at 11:59pm
# 
# **Note: There are two notebooks in Homework 2. Please also complete the other notebook `HW2_Linear_Models.ipynb` for full credit on this assignment.**

# ### Q4 : Decision Trees

# Download the dataset from this website : https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
# The data is a csv file with the following columns:
# 
# __age__: continuous.
# 
# __workclass__: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# 
# __fnlwgt__: continuous.
# 
# __education__: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# 
# __education-num__: continuous.
# 
# __marital-status__: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# 
# __occupation__: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# 
# __relationship__: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# 
# __race__: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# 
# __sex__: Female, Male.
# 
# __capital-gain__: continuous.
# 
# __capital-loss__: continuous.
# 
# __hours-per-week__: continuous.
# 
# __native-country__: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# 
# __target__: >50K, <=50K.

# 1. Read the data into a dataframe and assign column names

# In[54]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn import tree
from sklearn.model_selection import GridSearchCV

header_list = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
               "hours-per-week", "native-country", "target"]
read_file = pd.read_csv (r'/Users/Griffin/repos/assignment-2-klettg/adult.data')
read_file.to_csv (r'/Users/Griffin/repos/assignment-2-klettg/adult.csv', index=None)
df = pd.read_csv('/Users/Griffin/repos/assignment-2-klettg/adult.csv', names=header_list)
df.head(100)


# 2. Plot % of missing values in each column. Would you consider dropping any columns? Assuming we want to train a decision tree, would you consider imputing the missing values? If not, why?

# In[55]:


length = len(df)
feature_df = df.drop('target', 1)
pct_null = []
for col in feature_df:
    pct_null.append(round((((length - len(feature_df[feature_df[col] != ' ?'])) / length) * 100), 2))

    
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
features = header_list[:-1]
x_pos = [i for i, _ in enumerate(pct_null)]

plt.bar(x_pos, pct_null)
plt.xticks(x_pos, features, rotation='vertical')
plt.ylabel('% Missing Values')
plt.xlabel('Column Label')
plt.title('% Missing Values Across Columns')
plt.show()


# In[56]:


#2 Answer: I would consider dropping both workclass and occupation, since both features are missing > 5% of
#their values. I would not considering imputing values since all three with missing are categorical features
# and in a decision tree model we may split the tree off the imputed value. I will drop education-num
#since this contains the same information as the field education. Also
#I will drop native-country since almost all the values are the same
#and since there are so many categories, it would add a lot 
#of dimensionality with not so high information gain 

feature_df = feature_df.drop('education-num', 1)
feature_df = feature_df.drop('native-country', 1)


# 3. Pick 3 categorical features and for each categorical feature, plot side-by-side bars (horizontal or vertical) of class distribution for each category. 

# In[57]:


plt.figure()
count=0
columns_to_plot = ['workclass', 'occupation', 'race']
for col in columns_to_plot:
    df[col].value_counts().sort_index().plot(
        kind='bar', rot='vertical', ylabel='count',
        xlabel=col, title="count vs %s"%col)
    plt.show()


# 4. Split the dataset into development and test datasets using 80/20 ratio

# In[58]:


#Preprocessing
num_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

ohe_features = ['workclass','marital-status','occupation', 'relationship', 'sex', 'race']

te_features = ['education']

enc = OrdinalEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')
df[['target']] = enc.fit_transform(df[['target']])
df['target'] = df['target'].astype(int)

#scaler = StandardScaler()
#feature_df[num_features] = scaler.fit_transform(feature_df[num_features])
#feature_df[ohe_features] = ohe.fit_transform(feature_df[ohe_features])
#feature_df.head()
X_dev, X_test, y_dev, y_test = train_test_split(feature_df, df[['target']], random_state=42, test_size=0.2)


# 5. Fit a Decision Tree on the development data until all leaves are pure. What is the performance of the tree on development data and test data?

# In[6]:


from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from graphviz import Source
clf = DecisionTreeClassifier(random_state=81)
preprocess = make_column_transformer((StandardScaler(), num_features),
                                    (OneHotEncoder(handle_unknown='ignore'),ohe_features),
                                    (TargetEncoder(handle_unknown='ignore'),te_features),
                                     remainder="passthrough"
                                    )
#pipe = make_pipeline(preprocess,
#                   clf)
#pipe.fit(X_dev, y_dev)
#print(f"Test score:", pipe.score(X_test, y_test))
#print(f"Train score:", pipe.score(X_dev, y_dev))


pipe = make_pipeline(preprocess,
                    GridSearchCV(clf,
                                param_grid = {},
                                return_train_score=True))
  
pipe.fit(X_dev, y_dev)
grid_search_results = pipe.named_steps['gridsearchcv']
print(f"Best train score: ", grid_search_results.best_score_)
print(f"Best train alpha: ", grid_search_results.best_params_)
print(f"Test score:", pipe.score(X_dev, y_dev))
plt.figure(figsize=(30,50)) 
best_tree = grid_search_results.best_estimator_

ohe_feature_names = preprocess.named_transformers_["onehotencoder"].get_feature_names().tolist()
te_feature_names = preprocess.named_transformers_["targetencoder"].get_feature_names()
feature_names = num_features + ohe_feature_names + te_feature_names
target_values = [" <=50K", " >50K"]
tree_dot = plot_tree(best_tree, feature_names=feature_names, fontsize=9, filled=True, class_names=target_values)
plt.show()


# 6. Visualize the trained tree

# In[ ]:


print(tree.export_text(clf))


# 7. Prune the tree using one of the techniques discussed in class and evaluate the performance on the test set again.

# In[60]:


import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
clf = DecisionTreeClassifier(random_state=81)
preprocess = make_column_transformer((StandardScaler(), num_features),
                                    (OneHotEncoder(handle_unknown='ignore'),ohe_features),
                                    (TargetEncoder(handle_unknown='ignore'),te_features),
                                     remainder="passthrough"
                                    )
pipe = make_pipeline(preprocess,
                    GridSearchCV(clf,
                                param_grid = [{"min_impurity_decrease":np.logspace(-3,-1,100)}],
                                return_train_score=True))
  
pipe.fit(X_dev, y_dev)
grid_search_results = pipe.named_steps['gridsearchcv']
print(f"Best train score: ", grid_search_results.best_score_)
print(f"Best train alpha: ", grid_search_results.best_params_)
print(f"Test score:", pipe.score(X_test, y_test))
plt.figure(figsize=(16,25)) 
best_tree = grid_search_results.best_estimator_

ohe_feature_names = preprocess.named_transformers_["onehotencoder"].get_feature_names().tolist()
te_feature_names = preprocess.named_transformers_["targetencoder"].get_feature_names()
feature_names = num_features + ohe_feature_names + te_feature_names
target_values = [" <=50K", " >50K"]
tree_dot = plot_tree(best_tree, feature_names=feature_names, fontsize=10, filled=True, class_names=target_values)
plt.show()


# 8. List the top 3 most important features for this trained tree? How would you justify these features being the most important? 

# In[61]:


import seaborn as sns
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
ohe_feature_names = preprocess.named_transformers_["onehotencoder"].get_feature_names().tolist()
te_feature_names = preprocess.named_transformers_["targetencoder"].get_feature_names()
feature_names = num_features + ohe_feature_names + te_feature_names
feat_imps = zip(feature_names, best_tree.feature_importances_)
feats, imps = zip(*(sorted(list(filter(lambda x: x[1] != 0, feat_imps)), key=lambda x: x[1], reverse=True)))
ax = sns.barplot(list(feats), list(imps))
ax.tick_params(axis='x', rotation=90)
ax.set_ylabel('feature importance')
ax.set_xlabel('feature')
ax.set_title('feature importance across features')
#Answer:
#: Married/spouse status, education, capital-gain. We can see this using the decision tree's internal
# feature_importance as well as visually from the tree. These features are at the top of the tree, 
# which is gives us the highest impurity splits (most predictive of value)


# ### Q5: Random Forests

# Let's use the same dataset and the splits created in Q3.

# 1. Train a Random Forest model on the development dataset using RandomForestClassifier class in sklearn. Use the default parameters. Evaluate the performance of the model on test dataset. Does this perform better than Decision Tree on the test dataset (compare to results in Q 4.5)?

# In[15]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=81)
preprocess = make_column_transformer((StandardScaler(), num_features),
                                    (OneHotEncoder(handle_unknown='ignore'),ohe_features),
                                    (TargetEncoder(handle_unknown='ignore'),te_features),
                                     remainder="passthrough"
                                    )
pipe = make_pipeline(preprocess,
                    GridSearchCV(rf,
                                param_grid = [{}],
                                return_train_score=True))
  
pipe.fit(X_dev, y_dev)
grid_search_results = pipe.named_steps['gridsearchcv']
print(f"Best train score: ", grid_search_results.best_score_)
print(f"Best train alpha: ", grid_search_results.best_params_)
print(f"Test score:", pipe.score(X_test, y_test))

#ANSWER: Yes the random forest performs slightly better than the decision tree from 4.5
#with accuracy 86.3% vs 85.4%


# 2. Does all trees in the trained random forest model have pure leaves? How would you verify this?

# In[22]:


tree_dot = plot_tree(grid_search_results.best_estimator_.estimators_[0],
                filled=True)
plt.show()

#Yes all the trees in random forest have pure leaves. You can verify
#This by plotting individually one (or all) of the trees that comprise
#the random forest, or by looking at the default params on sklearn
#we see that the max_depth default is none, so nodes are expanded
#until all leaves are pure, and the default min_samples_split = 2 
#(so we would split until each leaf has one value if it is not pure yet)


# 3. Assume you want to improve the performance of this model. Also, assume that you had to pick two hyperparameters that you could tune to improve its performance. Which hyperparameters would you choose and why?

# In[ ]:


#I would search over different valuese for # of trees and # of features
#given to each tree. These values should vary based on our dataset sample
#size, and our specific dataset's number of features, so they make great
#candidates to have impact on overall performance


# 4. Now, assume you had to choose upto 10 different values (each) for these two hyperparameters. How would you choose these values that could potentially give you a performance lift? 

# In[62]:


#For this part, I would choose values spaced evenly larger and smaller 
#than the default hyperparameter values. If further optimization is required
#you could then perform a further search around the area that give you
#the best scores off of the first optimization. 


# 5. Perform model selection using the chosen values for the hyperparameters. Use cross-validation for finding the optimal hyperparameters. Report on the optimal hyperparameters. Estimate the performance of the optimal model (model trained with optimal hyperparameters) on test dataset? Has the performance improved over your plain-vanilla random forest model trained in Q5.1? 

# In[26]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=81)
preprocess = make_column_transformer((StandardScaler(), num_features),
                                    (OneHotEncoder(handle_unknown='ignore'),ohe_features),
                                    (TargetEncoder(handle_unknown='ignore'),te_features),
                                     remainder="passthrough"
                                    )
n_estimators = [ 50, 100, 150]
n_features = [4, 8, 12]

pipe = make_pipeline(preprocess,
                    GridSearchCV(rf,
                                param_grid = [{'n_estimators': n_estimators,
                                              'max_features': n_features}],
                                return_train_score=True,
                                n_jobs=2))
  
pipe.fit(X_dev, y_dev)
grid_search_results = pipe.named_steps['gridsearchcv']
print(f"Best train score: ", grid_search_results.best_score_)
print(f"Best train alpha: ", grid_search_results.best_params_)
print(f"Test score:", pipe.score(X_test, y_test))


# In[ ]:


#Best train score:  0.8572638600915514
#Best train alpha:  {'max_features': 8, 'n_estimators': 50}
#Test score: 0.8615077537233226


# 6. Can you find the top 3 most important features from the model trained in Q5.5? How do these features compare to the important features that you found from Q4.8? If they differ, which feature set makes more sense?

# In[27]:


best_rf = grid_search_results.best_estimator_
feat_imps = zip(feature_names, best_rf.feature_importances_)
feats, imps = zip(*(sorted(list(filter(lambda x: x[1] != 0, feat_imps)), key=lambda x: x[1], reverse=True)))
ax = sns.barplot(list(feats), list(imps))
ax.tick_params(axis='x', rotation=90)
ax.set_ylabel('feature importance')
ax.set_xlabel('feature')
ax.set_title('feature importance across features for best random forest')

#ANSWER: The features are different here for random forest - fnlwgt, age,and education. They are directionally similar, 
# with education, captial-gain, and married-spouse status all in top 5.
# I think these make less sense than the decision tree. Perhaps these are more important 
# because of how the random forest interactions took place, and how the features were split up when creating 
# the smaller trees within the forest. 


# ### Q6: Gradient Boosted Trees

# Let's use the same dataset and the splits created in Q3.

# 1. Choose three hyperparameters to tune GradientBoostingClassifier and HistGradientBoostingClassifier on the development dataset using 10-fold cross validation. Report on the time taken to do model selection for both the models. Also, report the performance of the test dataset from the optimal models.

# In[13]:


from sklearn.ensemble import GradientBoostingClassifier
learning_rate = [.01, .1, .2]
n_estimators = [50, 100, 200]
max_depth = [2,3,6]

gbc = GradientBoostingClassifier(random_state=81)
preprocess = make_column_transformer((StandardScaler(), num_features),
                                    (OneHotEncoder(handle_unknown='ignore'),ohe_features),
                                    (TargetEncoder(handle_unknown='ignore'),te_features),
                                     remainder="passthrough"
                                    )
pipe = make_pipeline(preprocess,
                    GridSearchCV(gbc,
                                param_grid = [{'learning_rate': learning_rate, 
                                              'n_estimators': n_estimators,
                                              'max_depth': max_depth}],
                                return_train_score=True,
                                cv=10,
                                n_jobs=2))
  
pipe.fit(X_dev, y_dev)
grid_search_results = pipe.named_steps['gridsearchcv']
print(f"Best train score: ", grid_search_results.best_score_)
print(f"Best train alpha: ", grid_search_results.best_params_)
print(f"Test score:", pipe.score(X_test, y_test))


# In[17]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import TransformerMixin

class DenseTransformer(TransformerMixin):
    def fit(self, X,y=None,**fit_params):
        return self
    
    def transform(self, X, y=None, **fit_params):
        return X.todense()

learning_rate = [.01, .1, .2]
n_estimators = [50, 100, 200]
max_depth = [2, 3, 6]

hgbc = HistGradientBoostingClassifier(random_state=81)
preprocess = make_column_transformer((StandardScaler(), num_features),
                                    (OneHotEncoder(handle_unknown='ignore'),ohe_features),
                                    (TargetEncoder(handle_unknown='ignore'),te_features),
                                     remainder="passthrough"
                                    )
pipe = make_pipeline(preprocess,
                     DenseTransformer(),
                    GridSearchCV(hgbc,
                                param_grid = [{'learning_rate': learning_rate, 
                                              'max_iter': n_estimators,
                                              'max_depth': max_depth}],
                                return_train_score=True,
                                cv=10,
                                n_jobs=2))
  
pipe.fit(X_dev, y_dev)
grid_search_results = pipe.named_steps['gridsearchcv']
print(f"Best train score: ", grid_search_results.best_score_)
print(f"Best train alpha: ", grid_search_results.best_params_)
print(f"Test score:", pipe.score(X_test, y_test))


# In[ ]:


#GradientBoost - Test score: 0.8757868877629357 - Model selection ~5 min
#HistGradientBoostingClassifier - Test score: 0.8780899738983571 - Model Selection ~<5 min 


# 2. Train an XGBoost model by tuning 3 hyperparameters using cross-validation. Report on the optimal hyperparameters and the time to train the model. Compare the performance of the trained XGBoost model on test dataset against the performances obtained from 6.1 

# In[18]:


from xgboost import XGBClassifier

eta = [.01, .1, .2]
max_depth = [3, 6, 9]
n_estimators = [50, 100, 150]

xgbc = XGBClassifier(random_state=81)
preprocess = make_column_transformer((StandardScaler(), num_features),
                                    (OneHotEncoder(handle_unknown='ignore'),ohe_features),
                                    (TargetEncoder(handle_unknown='ignore'),te_features),
                                     remainder="passthrough"
                                    )
pipe = make_pipeline(preprocess,
                    GridSearchCV(xgbc,
                                param_grid = [{'eta': eta, 
                                              'max_depth': max_depth,
                                              'n_estimators': n_estimators}],
                                return_train_score=True,
                                cv=10,
                                n_jobs=3))
  
pipe.fit(X_dev, y_dev)
grid_search_results_xgb = pipe.named_steps['gridsearchcv']
print(f"Best train score: ", grid_search_results_xgb.best_score_)
print(f"Best train alpha: ", grid_search_results_xgb.best_params_)
print(f"Test score:", pipe.score(X_test, y_test))


# In[ ]:


#Results: 
#Best train score:  0.871391584186148
#Best train alpha:  {'eta': 0.2, 'max_depth': 3, 'n_estimators': 150}
#Test score: 0.877475817595578


# 3. Compare the results on the test dataset from XGBoost, HistGradientBoostingClassifier, GradientBoostingClassifier with results from Q4.5 and Q5.5. Which model tends to perform the best and which one does the worst? How big is the difference between the two? Which model would you choose among these 5 models and why?

# In[15]:


# XGBOOST: Test score - 0.871391584186148
# HISTGRADIENTBOOSTINGCLASSIFIER: Test score - 0.8780899738983571
# GRADIENTBOOSTINGCLASSIFIER: Test score - 0.8757868877629357
# 4.5 Default Decision Tree: Test score - 0.8125002754349939
# 5.5 Optimized Random Forest: Test score - 0.8615077537233226
# HISTGRADIENTBOOSTINGCLASSIFIER seems to perform the best, but the margin is very very small between the boosting trees. 
# I would likely choose the histgradientboostingclassifier since it runs much faster than the other boosting algorithms
# and also has the best performance on the test set!


# 4. Can you list the top 3 features from the trained XGBoost model? How do they differ from the features found from Random Forest and Decision Tree? Which one would you trust the most?

# In[21]:


ohe_feature_names = preprocess.named_transformers_["onehotencoder"].get_feature_names().tolist()
te_feature_names = preprocess.named_transformers_["targetencoder"].get_feature_names()
feature_names = num_features + ohe_feature_names + te_feature_names
best_xgb = grid_search_results_xgb.best_estimator_
xgb_feat_imps = zip(feature_names, best_xgb.feature_importances_)
_xgb_feats, xgb_imps = zip(*(sorted(list(filter(lambda x: x[1] != 0, feat_imps)), key=lambda x: x[1], reverse=True)))
ax = sns.barplot(list(feats), list(imps))
ax.tick_params(axis='x', rotation=90)
ax.set_ylabel('feature importance')
ax.set_xlabel('feature')
ax.set_title('feature importance across features for best XGBoost')

#ANSWER: For XGB, Marital-Status (specifically, whether someone is married with civ spouse), Education, and capital-gain
# These are the exact same as the original decision tree top features (Married/spouse status, education, capital-gain)
# They differ from the Random Forest's (fnlwgt, age,and education)
# I would trust the original decision tree the most because of the interpretability still (you can show the tree)
# and explain the gini impurity difference to justify the importance


# 5. Can you choose the top 7 features (as given by feature importances from XGBoost) and repeat Q6.2? Does this model perform better than the one trained in Q6.2? Why or why not is the performance better?

# In[45]:


top_7_xgb_feats = feats[:7]
feature_names = num_features + ohe_feature_names + te_feature_names
for ele in feature_names:
    if ele  in top_7_xgb_feats:
        feature_names.remove(ele)

columns_to_drop = feature_names
print(columns_to_drop)


# In[53]:



print(top_7_xgb_feats)
from xgboost import XGBClassifier

eta = [.01, .1, .2]
max_depth = [3, 6, 9]
n_estimators = [50, 100, 150]
print(columns_to_drop)
xgbc_top_feat = XGBClassifier(random_state=81)
preprocess_xgbc_top_feat = make_column_transformer((StandardScaler(), num_features),
                                    (OneHotEncoder(handle_unknown='ignore'),ohe_features),
                                    (TargetEncoder(handle_unknown='ignore'),te_features),
                                    ("drop", columns_to_drop),
                                     remainder="passthrough"
                                    )
pipe_top_xgb = make_pipeline(preprocess_xgbc_top_feat,
                    GridSearchCV(xgbc_top_feat,
                                param_grid = [{'eta': eta, 
                                              'max_depth': max_depth,
                                              'n_estimators': n_estimators}],
                                return_train_score=True,
                                cv=10,
                                n_jobs=3,
                                verbose=True))
  
pipe_top_xgb.fit(X_dev, y_dev)
grid_search_results_xgb_top = pipe_top_xgb.named_steps['gridsearchcv']
print(f"Best train score: ", grid_search_results_xgb_top.best_score_)
print(f"Best train alpha: ", grid_search_results_xgb_top.best_params_)
print(f"Test score:", pipe_top_xgb.score(X_test, y_test))


# ### Q7: Calibration

# Let's use the same dataset and the splits created in Q3. Let's use the XGBoost model that you trained in Q6.2. 

# 1. Estimate the brier score for the XGBoost model (trained with optimal hyperparameters from Q6.2) scored on the test dataset. 

# In[64]:


from sklearn.metrics import brier_score_loss
xgb_best_estimator = grid_search_results_xgb.best_estimator_

#xgb_best_estimator.fit(X_dev,y_dev)
#print(xgb_best_estimator.feature_importances_)
#predictions = xgb_best_estimator.predict(X_test)
#brier_score = brier_score_loss(predictions, y_test)


# 2. Calibrate the trained XGBoost model using isotonic regression as well as Platt scaling. Plot predicted v.s. actual on test datasets from both the calibration methods

# In[ ]:


cal_svc = CalibratedClassifierCV(xgb_best_estimator, cv='prefit', method='sigmoid')
cal_svc.fit(X_calib, y_calib)
display = CalibrationDisplay.from_estimator(
    cal_svc, X_test, y_test, n_bins=10)

#For isotonic, same thing but use diff method w/in 
#CalibratedClassifierCV

cal_svc = CalibratedClassifierCV(xgb_best_estimator, cv='prefit', method='isotonic')
cal_svc.fit(X_calib, y_calib)
display = CalibrationDisplay.from_estimator(
    cal_svc, X_test, y_test, n_bins=10)
                                 


# 3. Report brier scores from both the calibration methods. Do the calibration methods help in having better predicted probabilities?

# In[ ]:


#code here


# In[ ]:




