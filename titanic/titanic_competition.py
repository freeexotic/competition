import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import numpy as np

train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")


train2 = train.dropna(axis=0, subset=['Survived'], inplace=True)
train2 = train.drop(axis=1, labels=['Name', 'Cabin', 'Ticket', 'Embarked'])
test2 = test.drop(axis=1, labels=['Name', 'Cabin', 'Ticket', 'Embarked'])


print(train2.isna().sum())

ordinc = OrdinalEncoder()
valid_train2 = train2.copy()
valid_test2 = test2.copy()
valid_train2[['Sex']] = ordinc.fit_transform(valid_train2[['Sex']])
valid_test2[['Sex']] = ordinc.transform(valid_test2[['Sex']])

imputer = SimpleImputer(strategy = 'mean')

valid_train2[['Age']] = pd.DataFrame(imputer.fit_transform(valid_train2[['Age']]))
valid_test2[['Age']] = pd.DataFrame(imputer.transform(valid_test2[['Age']]))

imputer2 = SimpleImputer(strategy = 'mean')

valid_test2[['Fare']] = pd.DataFrame(imputer2.fit_transform(valid_test2[['Fare']]))


y = valid_train2.Survived
X = valid_train2.drop(axis =1 , labels=['Survived'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)], eval_metric='logloss', verbose=False)

ans = my_model.predict(valid_test2)

for i in range(0, len(ans)):
  if ans[i] <= 0.5:
    ans[i] = int(0)
    ans[i] = int(ans[i])
  else:
    ans[i] = int(1)
    ans[i] = int(ans[i])

ans = ans.astype(int)

keys = pd.DataFrame({'PassengerId': valid_test2.PassengerId, 'Survived': ans})

keys.to_csv("titanic.csv", index=False)