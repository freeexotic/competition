
!pip install -q pandas_ta
!pip install catboost
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import pandas_ta as ta
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

sample_submission = pd.read_csv("path")
test = pd.read_csv("path")
train = pd.read_csv("path")

train.groupby('symbol').agg({"close":"max", "date":"max"}).reset_index()

test.groupby('symbol')['date'].min()

print(len(train['symbol'].unique()),"\n")
print(train.target.value_counts(), "\n")

train.info()

# препроцессинг данных

train['sample'] = 1 # note where we have train
test['sample'] = 0 # note where you have a test

train['id'] = 0
test['target'] = 0 # in case you don't have the value "target", we will continue the display anyway, because it's just polar zeros

data = pd.concat([train, test]).reset_index(drop=True) # concat

def gen_indicators(data, indicator_periods = [2, 3, 4, 6, 9, 12, 24, 48, 72, 96, 120, 168]):
        stock_data_tech = data.copy()
        stock_data_tech = stock_data_tech.sort_values(by='date')

        fe_df = pd.DataFrame()

        i=0

        for symbol in tqdm(stock_data_tech.symbol.unique()):
            data = stock_data_tech[stock_data_tech.symbol == symbol].copy()

            data.reset_index(inplace=True, drop=True)
            data.set_index('date', drop=False, inplace=True)

            
            for period in ([1,] + indicator_periods):
                data[f'pct_close_{period}'] = data['close'].pct_change(period)

            data.ta.pvol(append=True)
            data.ta.bop(append=True)

            for period in indicator_periods:

                if period > 2:
                    data.ta.willr(length=period, append=True)
                    data.ta.stoch(length=period, append=True)
                    data.ta.rsi(length=period, append=True)
                    data.ta.roc(length=period, append=True)
                    data.ta.mfi(length=period, append=True)
                    # data.ta.massi(length=period, append=True)  # EMA data leak !!!

            data.fillna(method="ffill", inplace=True)
            data.dropna(inplace=True)
            #data.fillna(method="bfill", inplace=True)
            #data.fillna(0, inplace=True)
            data.reset_index(inplace=True, drop=True)

            fe_df = pd.concat([fe_df, data], axis=0)
        fe_df.reset_index(inplace=True, drop=True)
        return fe_df

fe_data = gen_indicators(data)

fe_data

fe_train = fe_data.query('sample == 1').drop(['sample',], axis=1)
fe_test = fe_data.query('sample == 0').drop(['sample',], axis=1)

fe_test = fe_test.sort_values(by=['id',])

print(fe_test[['date', 'symbol', 'trades', 'id', 'target',]])

X_train = fe_train.drop(['open', 'high', 'low', 'close', 'volume', 'trades', 'symbol', 'date', 'id', 'target', 'close_unixtime'], axis=1)
y_train = fe_train.target.values

X_test = fe_test.drop(['open', 'high', 'low', 'close', 'volume', 'trades', 'symbol', 'date', 'id', 'target', 'close_unixtime'], axis=1)

model = CatBoostClassifier(verbose=100)
model.fit(X_train, y_train,)

feature_importances = model.get_feature_importance()

# Создаем DataFrame для удобства
features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Сортируем по убыванию важности
sorted_features_df = features_df.sort_values(by='Importance', ascending=False).head(25)

predict_proba = model.predict_proba(X_test)[:, 1]
sample_submission['target'] = predict_proba

def planka(x, i):
  if x >= i:
     x = 1
  else:
    x = 0
  return x

sample_submission['id'] = test['id']

sample_submission

sample_submission1 = pd.DataFrame(sample_submission.target.apply(lambda x: planka(x, 0.123)))

sample_submission1['id'] = test['id']

sample_submission1.to_csv('submission123.csv', index=False)

sample_submission['target'].hist(bins=250, edgecolor='black')
