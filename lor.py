import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
data = pd.read_csv("train.csv")
df=pd.DataFrame(data)
# sns.heatmap(data.isnull(),cbar=False,cmap='viridis')
# plt.show()
df.drop('Cabin',axis=1,inplace=True)
df.dropna(inplace=True)
sex=pd.get_dummies(df['Sex'],drop_first=True)
embark=pd.get_dummies(df['Embarked'],drop_first=True)
df=pd.concat([df,sex,embark],axis=1)
# print(df)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
X=df.drop('Survived',axis=1)
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
# print(X_train)
# logmodel=LogisticRegression()
logmodel = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=200, solver='saga'))
])
logmodel.fit(X_train,y_train)
pridictions=logmodel.predict(X_test)
# print(pridictions)
# print(classification_report(y_test,pridictions))
print(confusion_matrix(y_test,pridictions))