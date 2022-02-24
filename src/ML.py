from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np

import lightgbm as lgb
import seaborn as sns
import pickle


class MachineLearning:
    def __init__(self, model_file=None):
        self.model = None

        if model_file is not None:
            self.model = pickle.load(open(model_file, 'rb'))


    def learn(self, X, y, model_output_filename=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # params = {"n_estimators": np.arange(0,100,10),
        #    "max_depth": np.arange(2,1000,2),
        #    "num_leaves": np.arange(2,64,10),
        #    "learning_rate": [0.1, 0.25, 0.5],
        #    "random_state": [0]
        #     }

        """
        ベストパラメータ：{'subsample_freq': 7, 'subsample': 1.0, 'reg_lambda': 0.1, 'reg_alpha': 0.1, 'num_leaves': 28, 'min_child_samples': 0, 'max_depth': 8, 'learning_rate': 0.1, 'colsample_bytree': 0.9}
        ベストスコア:0.34351017839905695
        テストデータスコア
          MAE = 3599.230467237557
          MSE = 32515808.850873124
          RMSE = 5702.263484869243
          R2 = 0.30452972156647884

        """
        params = {'reg_alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
             'reg_lambda': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
             'num_leaves': [2, 4, 8, 10, 14, 20, 28, 32],
             'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
             "learning_rate": [0.1, 0.25, 0.5],
             "max_depth": [4, 8, 16, 32],
             'min_child_samples': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
             }            # 学習時fitパラメータ指定
        fit_params = {'verbose': 0,  # 学習中のコマンドライン出力
            'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
            'eval_metric': 'r2',  # early_stopping_roundsの評価指標
            'eval_set': [(X_train, y_train)]  # early_stopping_roundsの評価指標算出用データ
            }

        reg = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', n_estimators=10000)
        # reg = xgb.XGBRegressor(objective='reg:squarederror')
        k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
        # grid = GridSearchCV(estimator=reg, param_grid=params, cv=k_fold, scoring="r2", verbose=2)
        grid = RandomizedSearchCV(estimator=reg, param_distributions=params, scoring="r2", cv=k_fold, n_iter=1000, random_state=0, verbose=2)

        grid.fit(X_train, y_train, **fit_params)

        print(f"ベストパラメータ：{grid.best_params_}")
        print(f"ベストスコア:{grid.best_score_}")

        y_test_pred = grid.predict(X_test)
        y_test_pred = np.expand_dims(y_test_pred, 1)

        print("テストデータスコア")
        get_eval_score(y_test, y_test_pred)

        self.model = grid

        if model_output_filename is not None:
            pickle.dump(grid, open(model_output_filename, 'wb'))


    def predict(self, data):
        return self.model.predict(data)

    def score(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        y_test_pred = self.model.predict(X_test)
        y_test_pred = np.expand_dims(y_test_pred, 1)

        print("テストデータスコア")
        get_eval_score(y_test, y_test_pred)

    def get_eval_score(y_true,y_pred):

          mae = mean_absolute_error(y_true,y_pred)
          mse = mean_squared_error(y_true,y_pred)
          rmse = np.sqrt(mse)
          r2score = r2_score(y_true,y_pred)

          print(f"  MAE = {mae}")
          print(f"  MSE = {mse}")
          print(f"  RMSE = {rmse}")
          print(f"  R2 = {r2score}")
