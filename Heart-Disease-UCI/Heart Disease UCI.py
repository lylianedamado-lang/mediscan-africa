import numpy as np
import pandas as pd

from IPython.display import Image

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold,cross_validate,cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_curve, roc_auc_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
from subprocess import call

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.info()
df.head()
df.describe()

sns.heatmap(df.corr(), cmap='Blues')

X = df.drop(['target'],axis=1)
y = df['target']
X.head()
y.head()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

indep_scaler = StandardScaler()
X_train = indep_scaler.fit_transform(X_train)
X_test = indep_scaler.transform(X_test)

relatable_columns = ['trestbps', 'chol', 'thalach', 'oldpeak']
relatable_columns_index = [3, 4, 7, 9]

colors = ["skyblue", "olive", "gold", "teal"]

f, axes = plt.subplots(2, 2, figsize=(15, 15)) 
# sharex=True => if I use this parameter, then x- or y-axis will be shared among all subplots.

for index, each_column_index in enumerate(relatable_columns_index):
    print('\n{}. For {}({})'.format(index+1, df.columns[each_column_index], each_column_index))
    print("Kurtosis: %f" % pd.DataFrame(X_train)[each_column_index].kurt())
    sns.set(style="whitegrid")
    sns.boxplot(pd.DataFrame(X_train)[each_column_index],
                color=random.choice(colors), ax=axes[index//2, index%2])
    sns.swarmplot(pd.DataFrame(X_train)[each_column_index],color='black',alpha=0.25,
                 ax=axes[index//2, index%2])

f, axes = plt.subplots(2, 2, figsize=(15, 15))

for index, each_column_index in enumerate(relatable_columns_index):
    print('\n{}. For {}({})'.format(index+1, df.columns[each_column_index], each_column_index))
    print("Skewness: %f" % pd.DataFrame(X_train)[each_column_index].skew())
    sns.distplot( pd.DataFrame(X_train)[each_column_index] , color=random.choice(colors), ax=axes[index//2, index%2])

kfolds = 4 
split = KFold(n_splits=kfolds, shuffle=True, random_state=42)
base_models = [("DT_model", DecisionTreeClassifier(random_state=42)),
               ("RF_model", RandomForestClassifier(random_state=42,n_jobs=-1)),
               ("LR_model", LogisticRegression(random_state=42,n_jobs=-1)),
               ("XGB_model", XGBClassifier(random_state=42, n_jobs=-1))]
for name,model in base_models:
    clf = model
    cv_results = cross_val_score(clf, 
                                 X_train, y_train, 
                                 cv=split,
                                 scoring="accuracy",
                                 n_jobs=-1)
    min_score = round(min(cv_results), 4)
    max_score = round(max(cv_results), 4)
    mean_score = round(np.mean(cv_results), 4)
    std_dev = round(np.std(cv_results), 4)
    print(f"{name} cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")

rf = RandomForestClassifier(random_state=9)
rf.fit(X_train,y_train)

s = np.mean(cross_val_score(rf,X_train,y_train,scoring='roc_auc',cv=5))
print('The accuracy score for Random Forest is: ', s*100)

lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)

s = np.mean(cross_val_score(lr,X_train,y_train,scoring='roc_auc',cv=5))
print('The accuracy score for Logistic Regression is: ', s*100)

final_model = RandomForestClassifier(n_estimators=60, random_state=9, 
                                     criterion='gini', max_features='sqrt',
                                    max_samples=9)
final_model.fit(X_train,y_train)

s = np.mean(cross_val_score(final_model,X_train,y_train,scoring='roc_auc',cv=5))
print('The accuracy score for RandomForest is: ', s*100)

predictions = final_model.predict(X_test)
print('Testing Accuracy: ', accuracy_score(y_test,predictions))

# Classification Report
print('Classification Report :')
print(classification_report(y_test,predictions))

# Traing & Testing Accuracy
print("Training Accuracy :", final_model.score(X_train, y_train))
print("Testing Accuracy :", final_model.score(X_test, y_test))

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm,annot = True, annot_kws = {'size':15}, cmap = 'PuBu')

# Sensitivity & Specificity
total=sum(sum(cm))

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,1]+cm[0,1])
print('Specificity : ', specificity)

# ROC-AUC Curve Generating
print('The AUC Score: ',roc_auc_score(y_test, predictions))

y_scores = final_model.predict_proba(X_train)
y_scores = y_scores[:,1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
