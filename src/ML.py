from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from xfeat import SelectNumerical
from xfeat import ArithmeticCombinations, Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import lightgbm as lgb
import seaborn as sns
import pickle
from pycaret.classification import *

class MachineLearning:
    def __init__(self, model_file=None):
        self.model = None

        if model_file is not None:
            self.model = load_model(model_name=model_file)
            print(self.model)
            # self.model = pickle.load(open(model_file, 'rb'))

    def get_eval_score(self, y_true,y_pred):

          mae = mean_absolute_error(y_true,y_pred)
          mse = mean_squared_error(y_true,y_pred)
          rmse = np.sqrt(mse)
          r2score = r2_score(y_true,y_pred)

          print(f"  MAE = {mae}")
          print(f"  MSE = {mse}")
          print(f"  RMSE = {rmse}")
          print(f"  R2 = {r2score}")

    def learn(self, X, y, model_output_filename=None):
        mmscaler = MinMaxScaler(feature_range=(0, 2), copy=True)
        y['BUYSELL'] = mmscaler.fit_transform(y).astype('int')

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
        # params = {'reg_alpha': [0.1],
        #      'reg_lambda': [0.003],
        #      'num_leaves': [8],
        #      'colsample_bytree': [0.7],
        #      'subsample': [1.0],
        #      'subsample_freq': [2],
        #      "learning_rate": [0.1],
        #      "max_depth": [8],
        #      'min_child_samples': [3]
        #      }            # 学習時fitパラメータ指定

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
        cls = lgb.LGBMRegressor(
            boosting_type='gbdt',
            objective='multiclass',
            metric='multi_logloss',
            num_class=3,
            ).fit(X_train, y_train)

        y_train_pred = cls.predict(X_train)
        y_train_pred = np.argmax(y_train_pred, axis=1)

        y_test_pred = cls.predict(X_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        self.model = cls

        print(f"訓練データ正解率：{accuracy_score(y_train, y_train_pred)}")
        print(f"テストデータ正解率：{accuracy_score(y_test, y_test_pred)}")
        # reg = lgb.LGBMRegressor(boosting_type='gbdt', objective='', n_estimators=10000)
        # reg = xgb.XGBRegressor(objective='reg:squarederror')
        # k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
        # grid = GridSearchCV(estimator=reg, param_grid=params, cv=k_fold, scoring="r2", verbose=2)
        # grid = RandomizedSearchCV(estimator=reg, param_distributions=params, scoring="r2", cv=k_fold, n_iter=3, random_state=0, verbose=2)
        # grid.fit(X_train, y_train, **fit_params)
        # print(f"ベストパラメータ：{grid.best_params_}")
        # print(f"ベストスコア:{grid.best_score_}")
        # print(f"features: {X_train.columns}")
        # print(f"obj: {y_train.columns}")

        # y_test_pred = grid.predict(X_test)
        # y_test_pred = np.expand_dims(y_test_pred, 1)
        #
        # print("テストデータスコア")
        # self.get_eval_score(y_test, y_test_pred)
        # self.model = grid

        # feature importanceの確認
        # importance = pd.DataFrame(.feature_importances_, index=X_train.columns, columns=['importance'])
        # importance = importance.sort_values('importance', ascending=False)
        # print(f"importance < 4000 :{importance[importance['importance'] < 4000].index}")
        # print(f"全特徴量とimportance: \r\n{importance}")

        if model_output_filename is not None:
            pickle.dump(cls, open(model_output_filename, 'wb'))

        return accuracy_score(y_test, y_test_pred)


    def predict(self, data):
        return self.model.predict(data)

    def score(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        y_test_pred = self.model.predict(X_test)
        y_test_pred = np.expand_dims(y_test_pred, 1)

        print("テストデータスコア")
        self.get_eval_score(y_test, y_test_pred)

    def show_corr(self, data):
        data_corr = data.corr()
        print(data.corr())
        df = data_corr.sort_values('BUYSELL_PRICE')
        print(df['BUYSELL_PRICE'].head(15))
        print(df['BUYSELL_PRICE'].tail(10))
        x = len(data.columns)/1.5
        plt.figure(figsize=(x, x))
        sns.heatmap(data_corr, annot=True)
        plt.title("Corr Heatmap")
        plt.savefig("./corr.png", format="png")

    def reduce(self, data):
        # umapで2次元に削減
        reducer = umap.UMAP(random_state=42)
        reducer.fit(data)
        embedding = reducer.transform(data)

        # plot
        plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title('UMAP projection of the Digits dataset', fontsize=24);
        plt.savefig('umap.png', format="png")

    def feature_engineering(self, data):
        cols = data.columns.tolist()

        # 複数のカラムを組み合わせた特徴量を作成するパイプライン
        encoder = Pipeline([
            ArithmeticCombinations(input_cols=cols,
                                    drop_origin=False,
                                    operator="+",
                                    r=2,
                                    output_suffix="_plus"),
            ArithmeticCombinations(input_cols=cols,
                                    drop_origin=False,
                                    operator="*",
                                    r=2,
                                    output_suffix="_mul"),
            ArithmeticCombinations(input_cols=cols,
                                    drop_origin=False,
                                    operator="-",
                                    r=2,
                                    output_suffix="_minus"),
            ArithmeticCombinations(input_cols=cols,
                                    drop_origin=False,
                                    operator="+",
                                    r=3,
                                    output_suffix="_plus"),
        ])

        return encoder.fit_transform(data)

    def pycaret(self):
        df = pd.read_csv('./sampledata_200days_border001.csv', parse_dates=[0])
        df = df.set_index('Date')
        df = df.astype(float)
        df['BUYSELL'] = df['BUYSELL'].astype(int)
        df = df.dropna()
        df = df.drop(['BUYSELL_PRICE', 'Songiri', '_CLOSE_PCT_CHANGE', 'Coin', 'JPY'], 1) # 目的変数（BUYSELL_PRICE）以外の不要な変数を削除

        X = df.drop(['BUYSELL'], 1)
        y = df[['BUYSELL']]
        mmscaler = MinMaxScaler(feature_range=(0, 2), copy=True)
        y['BUYSELL'] = mmscaler.fit_transform(y).astype('int')

        ret = setup(df,
            target="BUYSELL",
            normalize=False,
            train_size=0.8,
            silent=True
            )
        compare_models(fold=10)
