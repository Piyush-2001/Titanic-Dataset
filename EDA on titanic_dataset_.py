import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

train = pd.read_csv('/content/train.csv')

train.head()

train.isnull()

sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='cividis')

train

sns.set_style('darkgrid')
sns.countplot(x="Survived", data=train)

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

sns.displot(train['Age'].dropna(),kde=True,color='skyblue')

sns.set_style('darkgrid')
sns.countplot(x="Pclass", data=train, palette="Set2")

sns.set_style('darkgrid')
sns.countplot(x="Sex", data=train, palette="Set3")

sns.set_style('darkgrid')
sns.countplot(x='Survived',hue='Embarked', data=train, palette="PuOr_r")

sns.set_style('darkgrid')
sns.countplot(x="SibSp", data=train, palette="cividis")

sns.set_style('darkgrid')
sns.countplot(x="Parch", data=train, palette="autumn")

sns.histplot(train['Fare'].dropna(),kde=True,color='darkred',binwidth=10,bins=30)

plt.figure(figsize=(15, 8))
sns.boxplot(x='Pclass',y='Age',data=train,palette='pastel')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='cividis')

train.drop('Cabin',axis=1,inplace=True)

train.head()

train.dropna(inplace=True)

train

train.info()

sex = pd.get_dummies(train['Sex'],drop_first=True)
sex.head()

embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()

train = train.drop(['Sex','Embarked','Name','Ticket'],axis=1)
train.head()

train = pd.concat([train,sex,embark],axis=1)
train.head()

train.drop('Survived',axis=1).head()

train['Survived'].head()

from sklearn.model_selection import train_test_split

trainx = train.drop('Survived',axis=1)
trainy = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(trainx,trainy,test_size=0.30,random_state=101)

"""# Logistic Regression"""

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

log_pred = logmodel.predict(X_test)

accuracy=confusion_matrix(y_test,log_pred)

accuracy

accuracy=accuracy_score(y_test,log_pred)
accuracy
#0.8014981273408239

print(classification_report(y_test,log_pred))

"""## Cross Validation"""

model_cv = LogisticRegressionCV(10)
model_cv.fit(X_train, y_train)

model = LogisticRegression(C=10)
scores = cross_val_score(model, X_train, y_train, cv=5)
scores

scores.mean()
#0.7733161290322581

"""# Decision Tree Classifier"""

clf = DecisionTreeClassifier(random_state=42,max_depth=3)
clf.fit(X_train,y_train)

plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True)

dt_pred = clf.predict(X_test)

confusion_matrix = confusion_matrix(y_test,dt_pred)
matrix_df = pd.DataFrame(confusion_matrix)

ax = plt.axes()
sns.set(font_scale=1.3)
plt.figure(figsize=(10,7))
sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")

ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize =15)
ax.set_xticklabels([''])
ax.set_ylabel("True Label", fontsize=15)
ax.set_yticklabels(list(trainy.unique()), rotation = 0)
plt.show()

accuracy_score(y_test,dt_pred)
#0.846441947565543

print(classification_report(y_test,dt_pred))

"""# Random Forest

"""

X, y = make_classification(n_features=9,random_state=42, shuffle=True)

rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
rf_clf.fit(X_train, y_train)

rf_pred = clf.predict(X_test)

accuracy_score(y_test,rf_pred)
#0.846441947565543

print(classification_report(y_test,rf_pred))
