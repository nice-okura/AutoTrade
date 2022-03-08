import math
from datetime import datetime
from datetime import timedelta
import os
import sys
import csv
import pandas as pd
import numpy as np
import logging
from CryptService import CryptService
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pprint import pprint as pp
import talib
from ML import MachineLearning
import itertools

DEBUG = True

URL = "https://api.bitbank.cc/v1"
PUBLIC_URL = "https://public.bitbank.cc"
BUY = 1
SELL = -1

# 各種パラメータ
class Parameter:
    def __init__(self,
                 ma_short=5,
                 ma_lomg=50,
                 candle_type='1hour',
                 pair='qtum_jpy',
                 ma_times=12,
                 buy_price=400.0,
                 sell_price=400.0,
                 rsi_sell=80.0,
                 rsi_buy=20.0,
                 vol_order=200000,
                 weight_of_price=0.05, # 連続MA回数から購入金額を決めるときの重み
                 logic=0,
                 price_decision_logic=0,
                 songiri=True,
                 songiri_perc=0.1, # 損切する価格変動ボーダー
                 ml_model=None):

      self.MA_short = ma_short  # 移動平均（短期）
      self.MA_long = ma_lomg  # 移動平均（長期）
      self.CANDLE_TYPE = candle_type  # データ取得間隔
      self.PAIR = pair # 対象通貨
      self.MA_times = ma_times  # コインを購入/売却金額する連続GC/DC回数
      self.BUY_PRICE = buy_price  # 購入金額(円)
      self.SELL_PRICE = sell_price # 売却金額(円)
      self.RSI_SELL = rsi_sell  # 売りRSIボーダー
      self.RSI_BUY = rsi_buy # 買いRSIボーダー
      self.VOL_ORDER = vol_order  # 取引する基準となる取引量(Volume)
      self.WEIGHT_OF_PRICE = weight_of_price
      self.LOGIC = logic # 売り買いロジック
      self.PDL = price_decision_logic # 売買価格決定ロジック
      self.SONGIRI = songiri # 損切実施するかどうか(True: 損切する, False: 損切しない)
      self.SONGIRI_PERC = songiri_perc # 損切する価格変動ボーダー
      self.ML_MODEL = ml_model # 機械学習モデルのファイル名


class AutoTrade:

    def __init__(self, param):
        self.param = param
        self.cs = CryptService(URL, PUBLIC_URL, os.environ['API_KEY'], os.environ['API_SECRET'], "bitbank")
        self.ml = None
        # 全特徴量
        # self.features = ["Open", "High", "Low", "Close", "Volume", "ma_diff", "GCDC_times", "BBANDS_upperband", "BBANDS_middleband", "BBANDS_lowerband", "MA_SHORT", "MA_LONG", "MIDPOINT", "MACD_macd", "MACD_macdsignal", "MACD_macdhist", "RSI", "OBV", "ATR", "STDDEV", "DEMA", "EMA", "HT_TRENDLINE", "KAMA", "SMA", "T3", "TEMA", "TRIMA", "WMA", "ADX", "ADXR", "APO", "AROON_aroondown", "AROON_aroonup", "AROONOSC", "BOP", "CCI", "DX", "MFI", "MINUS_DI", "MINUS_DM", "MOM", "PLUS_DI", "PLUS_DM", "STOCH_slowk", "STOCH_slowd", "STOCHF_fastk", "STOCHF_fastd", "STOCHRSI_fastk", "STOCHRSI_fastd", "TRIX", "ULTOSC", "WILLR", "AD", "ADOSC", "NATR", "TRANGE", "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR_inphase", "HT_PHASOR_quadrature", "HT_SINE_sine", "HT_SINE_leadsine", "HT_TRENDMODE", "BETA", "CORREL", "LINEARREG", "LINEARREG_ANGLE", "LINEARREG_INTERCEPT", "LINEARREG_SLOPE", "JPY", "Coin"]
        # High Importance features
        self.features = ['CORREL', 'BBANDS_lowerband', 'BBANDS_upperband', 'MFI', 'MINUS_DI', 'ADOSC', 'ULTOSC', 'DEMA', 'HT_SINE_sine', 'ADX', 'STOCH_slowd','STOCHF_fastk', 'STOCHRSI_fastd', 'HT_SINE_leadsine', 'BBANDS_middleband', 'TEMA', 'RSI', 'MOM', 'MINUS_DM', 'GCDC_times',  'ADXR', 'JPY', 'STOCH_slowk', 'HT_DCPHASE', 'WILLR', 'PLUS_DM', 'MACD_macdhist', 'KAMA', 'MIDPOINT', 'TRIX', 'APO', 'CCI', 'Coin', 'OBV', 'AD', 'LINEARREG_ANGLE', 'ATR', 'ma_diff', 'LINEARREG_INTERCEPT', 'MA_LONG', 'T3', 'MACD_macdsignal', 'EMA', 'MACD_macd', 'HT_TRENDLINE', 'Open', 'SMA', 'WMA', 'STOCHRSI_fastk', 'AROONOSC', 'TRIMA', 'AROON_aroondown', 'AROON_aroonup', 'Low', 'High', 'Close', 'HT_TRENDMODE', 'MA_SHORT', 'STOCHF_fastd', 'LINEARREG_SLOPE']
        # self.features = ['BOP', 'HT_PHASOR_quadrature', 'STDDEV', 'BETA', 'HT_PHASOR_inphase', 'TRANGE', 'LINEARREG', 'Volume', 'DX', 'NATR', 'PLUS_DI', 'HT_DCPERIOD', 'CORREL', 'BBANDS_lowerband', 'BBANDS_upperband', 'MFI', 'MINUS_DI', 'ADOSC', 'ULTOSC', 'DEMA', 'HT_SINE_sine', 'ADX', 'STOCH_slowd', 'STOCHF_fastk', 'STOCHRSI_fastd', 'HT_SINE_leadsine', 'BBANDS_middleband', 'TEMA', 'RSI', 'MOM', 'MINUS_DM', 'GCDC_times', 'ADXR']

        # self.features = ["MA_LONG", "EMA", "MA_SHORT", "BBANDS_middleband", "BBANDS_lowerband", "HT_TRENDLINE"]
        # self.features = ["BBANDS_upperband", "BBANDS_lowerband", "MA_SHORT", "MA_LONG", "MIDPOINT", "MACD_macdsignal", "RSI", "OBV", "ATR", "STDDEV", "STOCH_slowk", "STOCH_slowd", "STOCHRSI_fastk", "STOCHRSI_fastd", "Coin", "JPY"]
        # self.features = ["RSI", "MACD_macd", "ma_diff", "MACD_macdsignal", "STOCH_slowk", "STOCH_slowd", "MACD_macdhist", "Close", "Low", "High", "Open", "STOCHRSI_fastk", "STOCHRSI_fastd"]

        # 1. ロガーを取得する
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # 出力レベルを設定

        # 2. ハンドラーを生成する
        h = logging.StreamHandler()
        h.setLevel(logging.DEBUG)  # 出力レベルを設定
        h2 = logging.FileHandler('./AutoTrade.log')
        if DEBUG == True:
            h2.setLevel(logging.DEBUG)  # 出力レベルを設定
        else:
            h2.setLevel(logging.INFO)  # 出力レベルを設定

        # 3. フォーマッタを生成する
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 4. ハンドラーにフォーマッターを設定する
        h.setFormatter(fmt)
        h2.setFormatter(fmt)

        # 5. ロガーにハンドラーを設定する
        logger.addHandler(h)
        logger.addHandler(h2)

        self.logger = logger

    # 特徴量の計算
    def calc_features(self, df):
        open = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']

        orig_columns = df.columns

        hilo = (df['High'] + df['Low']) / 2
        df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        df['BBANDS_upperband'] -= hilo
        df['BBANDS_middleband'] -= hilo
        df['BBANDS_lowerband'] -= hilo
        df['DEMA'] = talib.DEMA(close, timeperiod=30) - hilo
        df['EMA'] = talib.EMA(close, timeperiod=30) - hilo
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close) - hilo
        df['KAMA'] = talib.KAMA(close, timeperiod=30) - hilo
        df['MA_SHORT'] = talib.MA(close, timeperiod=self.param.MA_short, matype=0) - hilo
        df['MA_LONG'] = talib.MA(close, timeperiod=self.param.MA_long, matype=0) - hilo
        df['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
        df['SMA'] = talib.SMA(close, timeperiod=30) - hilo
        df['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
        df['TEMA'] = talib.TEMA(close, timeperiod=30) - hilo
        df['TRIMA'] = talib.TRIMA(close, timeperiod=30) - hilo
        df['WMA'] = talib.WMA(close, timeperiod=30) - hilo

        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
        df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
        df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(high, low, timeperiod=14)
        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
        df['BOP'] = talib.BOP(open, high, low, close)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        df['DX'] = talib.DX(high, low, close, timeperiod=14)
        df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        # skip MACDEXT MACDFIX たぶん同じなので
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
        df['MOM'] = talib.MOM(close, timeperiod=10)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
        df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        df['TRIX'] = talib.TRIX(close, timeperiod=30)
        df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        df['AD'] = talib.AD(high, low, close, volume)
        df['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        df['OBV'] = talib.OBV(close, volume)

        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
        df['TRANGE'] = talib.TRANGE(high, low, close)

        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
        df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
        df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
        df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close)
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

        df['BETA'] = talib.BETA(high, low, timeperiod=5)
        df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
        df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14) - close
        df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
        df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
        df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
        df['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)
        df['_CLOSE_PCT_CHANGE'] = -close.pct_change(-5) # ５単位時間後との価格差

        return df


    # OHLCVの取得
    def get_ohlcv(self, date, size, candle_type):
        """dateからsize単位時間分のOHLCVデータを取得する

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO

        """

        ohlcv_df = pd.DataFrame()

        # 必要なチャートを取得
        while(len(ohlcv_df) < size):
            d_str = date.strftime('%Y%m%d')
            self.logger.debug(d_str + "分 データ取得開始")
            cd = self.cs.get_candlestick(candle_type, self.param.PAIR, d_str)

            if cd['success'] == 0:
                self.logger.info(d_str + "分 データなし")
                self.logger.info(cd)

                date = date - timedelta(days=1)
                continue

            ohlcv = self.cs.get_candlestick(candle_type, self.param.PAIR, d_str)["data"]["candlestick"][0]["ohlcv"]
            columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']

            df_new = pd.DataFrame(data=ohlcv, columns=columns)
            df_new['Date']=pd.to_datetime(df_new['Date'], unit='ms', utc=True)
            df_new = df_new.set_index('Date')
            df_new = df_new.astype(float)
            df_new.index = df_new.index.tz_convert('Asia/Tokyo')

            ohlcv_df = pd.concat([df_new, ohlcv_df])
            date = date - timedelta(days=1)

        return ohlcv_df


    # CSVからOHLCVを取得し、CSVにする
    def load_csv2pd(self, filename):
        """CSVファイルからDataFrameを読み込み、返す

        """
        df = pd.read_csv(filename, parse_dates=[0])
        df = df.set_index('Date')
        df = df.astype(float)

        return df


    # 移動平均の差(ma_diff)と連続GC/DC回数を計算し、返却
    def get_madata(self, df):
        ma_short = df.rolling(self.param.MA_short).mean()
        ma_long = df.rolling(self.param.MA_long).mean()
        ma_diff = ma_short['Close'] - ma_long['Close']
        # 連続GC/DC回数を計算
        times_list = []
        for i, d in enumerate(ma_diff):
            # 以下のいずれか場合は連続GC/DC回数を初期値（1）に設定
            # 　初めのデータ：i == 0
            #　 差分データがないとき：math.isnan(d)
            # 　前回データがないとき：math.isnan(df['ma_diff'][i-1])
            # 　GC -> DCまたはDC -> GCのとき：d * df['ma_diff'][i-1] < 0
            if i == 0 or math.isnan(d) or math.isnan(ma_diff[i-1]) or d * ma_diff[i-1] < 0 :
                times_list.append(1)
            elif d is not np.nan:
                times_list.append(times_list[i-1]+1)

        return ma_diff, times_list


    # 連続times x n回 DCまたはGCを継続しているか判定
    def is_gcdc(self, df, times):
        return df['GCDC_times'][-1] % times == 0


    # RSIをもとに売り(-1) or 買い(1) or ステイ(0)を判定
    def buysell_by_rsi(self, df):
        buysell = 0

        if df['RSI'][-1] <= self.param.RSI_BUY:
            buysell = BUY
        elif df['RSI'][-1] >= self.param.RSI_SELL:
            buysell = SELL

        return buysell


    # Volumeをもとに取引するかしないを判定
    def buysell_by_vol(self, df):
        return df['Volume'][-1] >= self.param.VOL_ORDER


    # 売買
    def order(self, buysell, price_yen, coin_price):
        """
        コインを購入もしくは売却
        """
        price = price_yen/coin_price
        order_mode, order_str = ("buy", "購入") if buysell == BUY else ("sell", "売却")

        if DEBUG is False:
            order_result = self.cs.post_order(self.param.PAIR, price, "", order_mode, "market")
        else:
            order_result = {'success': 0, 'data': "デバッグモード"}
        if order_result['success'] == 1:
            # オーダー成功
            price = order_result['data']['start_amount']
            self.logger.info(self.param.PAIR + "を" + str(price_yen) + "円で" + str(price) + order_str)

            return price
        else:
            # オーダー失敗
            self.logger.error("オーダー失敗")
            self.logger.error(order_result)

            return -1

        return -1


    def buyORsell(self, df):
        """売りか買いか判定

        Parameters
        -----------
        df : DataFrame
        logic : int
            0 : デフォルトロジック
            1 : ・・・

        Return
        -----------
        buysell : int
            BUY : 買い
            SELL : 売り
            0 : ステイ
        """
        buysell = 0
        logic = self.param.LOGIC

        if 'ma_diff' in df.columns:
            gcdc = "GC" if df['ma_diff'][-1] >= 0 else "DC"
        else:
            self.logger.warning("ma_diff カラムがありません")
            return buysell

        """
            メインの売り買いロジック
        """
        if self.ml is not None:
            """
            機械学習のモデルを読み込んで売り買い判定する
            """
            df = df[self.features]
            # df = df.drop(['BUYSELL', '_CLOSE_PCT_CHANGE', 'SimulateAsset', 'Profit', 'Coin', 'JPY', 'Songiri'], axis=1)
            pred_df = self.ml.predict(df[-1:])
            yen = df['JPY'][-1]
            border = yen*0.05
            if int(pred_df[0]) < -border:
                buysell = SELL
            elif int(pred_df[0]) > border:
                buysell = BUY

        elif logic == 0:
            if self.buysell_by_vol(df):
                if (self.is_gcdc(df, self.param.MA_times) and gcdc == "GC") or self.buysell_by_rsi(df) == BUY:
                    buysell = BUY
                elif (self.is_gcdc(df, self.param.MA_times) and gcdc == "DC") or self.buysell_by_rsi(df) == SELL:
                    buysell = SELL

        elif logic == 1:
            """
            売り買いロジック③：
            　MAのみで判断。連続n回GCなら「買い」、連続n回DCなら「売り」
            """
            if self.is_gcdc(df, self.param.MA_times) and gcdc == "GC":
                buysell = BUY
            elif self.is_gcdc(df, self.param.MA_times) and gcdc == "DC":
                buysell = SELL

        elif logic == 2:
            """
            売り買いロジック⑪：
            Volumeがxxx以上でないと売買しない
            　優先度１．RSIで判断
            　　RSI売られすぎのときに「買い」、買われすぎのときに「売り」
            　優先度２．移動平均で判断
            　　連続n回GCなら「買い」、連続n回DCなら「売り」
            """
            if self.buysell_by_vol(df):
                if self.buysell_by_rsi(df) == BUY:
                    buysell = BUY
                elif self.buysell_by_rsi(df) == SELL:
                    buysell = SELL
                else:
                    if self.is_gcdc(df, self.param.MA_times) and gcdc == "GC":
                        buysell = BUY
                    elif self.is_gcdc(df, self.param.MA_times) and gcdc == "DC":
                        buysell = SELL
        elif logic == 3:
            """
            RSIだけで判断
            """
            if self.buysell_by_rsi(df) == BUY:
                buysell = BUY
            elif self.buysell_by_rsi(df) == SELL:
                buysell = SELL

        elif logic == 4:
            """
            STOCH RSIで判断
            """
            price = df['Close'][-1]
            k = df['STOCH_slowk'][-1]
            d = df['STOCH_slowd'][-1]
            delta = 10.0

            if k >= self.param.RSI_SELL and d >= self.param.RSI_SELL:
                if 0.0 < k - d < delta:
                    buysell = SELL
                    # print(f"[SELL] {price=} {k=} {d=}")

            elif k <= self.param.RSI_BUY and d <= self.param.RSI_BUY:
                if 0.0 < d - k < delta:
                    buysell = BUY
                    # print(f"[BUY] {price=} {k=} {d=}")
        elif logic == 10:
            """
            XRP円平均法で定期的に購入する
            """
            hour = int(df.index[-1].strftime('%H'))

            if hour == 0 or hour == 12:
                """
                0時または12時の時に購入
                """
                buysell = BUY

        elif logic == -1:
            """
            （注意：未来データを使ったテスト指標なので実際には使えない）
            ５単位時間後にどのくらい価格変化するかを表した指標（_CLOSE_PCT_CHANGE）が
            border%以上変化する場合、売買する。
            """
            border = 0.05

            if df['_CLOSE_PCT_CHANGE'][-1] >= border:
                buysell = BUY
            elif df['_CLOSE_PCT_CHANGE'][-1] <= -border:
                buysell = SELL

        else:
            self.logger.error("対応ロジックなし logic: " + logic)

        return buysell


    def fee(self, buysell_price):
        return buysell_price*0.0002


    def get_BUYSELLprice(self, yen_price, coin_price, coin, jpy, oneline_df=None):
        """ 売買価格を決める

        """
        BUYSELLprice = 0.0
        price_decision_logic = self.param.PDL

        if self.ml is not None:
            """
            機械学習のモデルを読み込んで売り買い判定する
            """
            oneline_df = oneline_df[self.features]
            pred_df = self.ml.predict(oneline_df[-1:])
            BUYSELLprice = abs(int(pred_df))
            # BUYSELLprice = 1000

        elif price_decision_logic == 0:
            """
            パラメータで設定した価格で一律売買（重みづけなどなし）
            """
            BUYSELLprice = yen_price

        elif price_decision_logic == 1 and oneline_df is not None:
            """
            連続GC/DC回数と、WEIGHT_OF_PRICE(重み付け)から売り買い価格を決める
            """
            BUYSELLprice = yen_price * oneline_df['GCDC_times'][0] * self.param.WEIGHT_OF_PRICE
            # BUYSELLprice = yen_price * np.log10(oneline_df['GCDC_times'][0])
        elif price_decision_logic == 2 and oneline_df is not None:
            """
            連続GC/DC回数と、WEIGHT_OF_PRICE(重み付け)から売り買い価格を決める
            売り価格は所持仮想通貨数(coin)、買い価格は所持日本円(jpy)から決める
            """
            gcdt_times = oneline_df['GCDC_times'][0]
            weight = (gcdt_times/self.param.MA_times * self.param.WEIGHT_OF_PRICE)*0.3
            #self.logger.debug(f"{weight=:.2%}")
            ma_diff = oneline_df['ma_diff'][0]

            if ma_diff < 0:
                # 売り
                BUYSELLprice = coin*coin_price*weight
            elif ma_diff > 0:
                # 買い
                BUYSELLprice = jpy*weight

        elif price_decision_logic == -1:
            """
            n単位時間後の価格変化率(pct_chg)から売り買い価格を決める
            売り価格は所持仮想通貨数(coin)、買い価格は所持日本円(jpy)から決める
            """
            pct_chg = oneline_df['_CLOSE_PCT_CHANGE'][0]
            if pct_chg < 0:
                # 売り
                # BUYSELLprice = coin_price*np.abs(pct_chg*2.0)
                BUYSELLprice = coin*coin_price*np.abs(pct_chg*2.0)
            elif pct_chg > 0:
                # 買い
                BUYSELLprice = jpy*np.abs(pct_chg*2.0)
                # BUYSELLprice = jpy*np.abs(pct_chg*2.0)

        return BUYSELLprice


    def check_minus(self, df):
        minus_coin = df.query('Coin < 0')
        minus_jpy = df.query('JPY < 0')

        return_status = 0
        if len(minus_coin) != 0:
            print(f"Error: 所持コイン数: {minus_coin['Coin'].iloc[-1]}") # iloc[-1]でSeries型の値のみ取り出す
            return_status = -1

        if len(minus_jpy) != 0:
            print(f"Error: 所持日本円: {minus_jpy['JPY'].iloc[-1]}")
            return_status = -1

        return return_status


    def songiri(self, df, position_df, coin_price, coin, yen, tmp_df):
        """ 今の価格にて、これまでの売り買いポジションから、perc%以上の損失が出ている場合、
            ポジションを解放（売りポジなら買い、買いポジなら売り）する

        Paramters
        ------------
        df : DataFrame : これまでの価格情報
        position_df : DataFrame : ポジションリスト
        coin_price : float : 現在のコインの価格
        coin : float : 所持コイン数
        yen : float : 所持日本円
        tmp_df : DataFrame : 最新の価格情報

        Return
        ------------
        df : DataFrame : 損切した場合の売り買い情報が反映されたdef
        position_df : DataFrame : ポジション解放を反映させたもの
        """
        perc = self.param.SONGIRI_PERC
        i = tmp_df.index[0]
        # print(f"tmp_df  Date: {i.strftime('%Y/%m/%d %H:%M:%S')} coin_price:{coin_price}")

        for j, p in position_df.iterrows():
            """
             j: 当時売買した日時
            """
            # if p['BUYSELL'] == BUY:
            #     print(f"position BUY  Date: {j.strftime('%Y/%m/%d %H:%M:%S')} Close:{p['Close']} border:{p['Close']*(1-perc)}")
            # elif p['BUYSELL'] == SELL:
            #     print(f"position SELL  Date: {j.strftime('%Y/%m/%d %H:%M:%S')} Close:{p['Close']} border:{p['Close']*(1+perc)}")
            # sys.exit()

            # print(f"border: {p['Close']*(1+perc)}")
            # 購入ポジションがあり、現在の価格(coin_price)が購入価格(p['Close'])からperc%以上下がっている場合、売る
            if p['BUYSELL'] == BUY and coin_price <= p['Close']*(1-perc):
                print(f"  {p.name.strftime('%Y/%m/%d %H:%M:%S')}に{p['Close']}円で買ったものを{i}に{coin_price}で売る")

                df.at[i, 'BUYSELL'] = SELL
                df.at[i, 'Songiri'] = True

                sell_price = self.get_BUYSELLprice(self.param.SELL_PRICE, coin_price, coin, yen, oneline_df=tmp_df)  # 購入する仮想通貨の枚数
                # print(f"  SELL Price: {sell_price}")
                yen += sell_price
                coin -= sell_price/coin_price
                position_df = position_df.drop(j)
                break

            elif p['BUYSELL'] == SELL and coin_price >= p['Close']*(1+perc):
                print(f"  {p.name.strftime('%Y/%m/%d %H:%M:%S')}に{p['Close']}円で売ったものを{i}に{coin_price}で買う")

                # print(f"{coin_price=} ")
                df.at[i, 'BUYSELL'] = BUY
                df.at[i, 'Songiri'] = True

                buy_price = self.get_BUYSELLprice(self.param.BUY_PRICE, coin_price, coin, yen, oneline_df=tmp_df)  # 購入する仮想通貨の枚数
                # print(f"  BUY Price: {buy_price}")
                yen -= buy_price
                coin += buy_price/coin_price
                position_df = position_df.drop(j)
                break

        return df, position_df, coin, yen


    def simulate(self, df, init_yen=100000.0, init_coin=100.0):
        # self.logger.debug("## simulate ")
        """
            過去データ(df)から実際に売買した場合の総資産や利益を計算し、dfに追加して返す

        Parameters
        ----------
        logic : int
            0 : デフォルトロジック
            1 : ・・・
            -1 : n単位時間後にどのくらい価格変化するかを表した指標（実際には使えない）
        init_yen : int 初期日本円
        init_coin : int 初期仮想通貨数
        price_decision_logic : int 売買決定決定ロジック

        """
        yen = init_yen  # 初期日本円
        coin = init_coin  # 初期仮想通貨数
        init_asset = init_yen + init_coin * df['Close'][0]
        df['BUYSELL'] = 0             # 売り買いの識別　index 8
        df['SimulateAsset'] = 0.0     # シミュレーションしたときの総資産　index 9
        df['Profit'] = 0.0            # シミュレーションしたときの利益（総資産ー初期資産）index 10
        df['Coin'] = init_coin        # 所持仮想通貨数　index 11
        df['JPY'] = init_yen
        df['Songiri'] = False
        df['BUYSELL_PRICE'] = 0

        logic = self.param.LOGIC

        position_df = pd.DataFrame()

        for i, r in df.iterrows():
            tmp_df = pd.DataFrame([r])
            coin_price = tmp_df['Close'][0]  # 購入する仮想通貨の現在の価格

            pct_chg = tmp_df['_CLOSE_PCT_CHANGE'][0]

            if self.buyORsell(tmp_df) == BUY:
                """
                購　入
                """
                buy_price = self.get_BUYSELLprice(self.param.BUY_PRICE, coin_price, coin, yen, oneline_df=tmp_df)  # 購入する仮想通貨の枚数

                if yen > buy_price:
                    df.at[i, 'BUYSELL'] = BUY
                    yen -= (buy_price + self.fee(buy_price))
                    coin += buy_price/coin_price
                    #self.logger.debug(f'[BUY]{tmp_df.index.strftime("%Y/%m/%d %H:%M")[0]}: BUY_PRICE: {buy_price:.2f} {coin=:.2f}')
                    #self.logger.debug(f'   PCT_CHG:{pct_chg:.2%} jpy:{yen}')
                    df.at[i, 'BUYSELL_PRICE'] = buy_price
                else:
                    print(f"{buy_price-yen:.0f}円 不足")

            elif self.buyORsell(tmp_df) == SELL:
                """
                売　却
                """
                sell_price = self.get_BUYSELLprice(self.param.SELL_PRICE, coin_price, coin, yen, oneline_df=tmp_df)  # 購入する仮想通貨の枚数

                if coin > sell_price/coin_price:
                    df.at[i, 'BUYSELL'] = SELL
                    yen += (sell_price - self.fee(sell_price))
                    coin -= sell_price/coin_price
                    #self.logger.debug(f'[SELL]{tmp_df.index.strftime("%Y/%m/%d %H:%M")[0]}: SELL_PRICE: {sell_price:.2f} {coin=:.2f}')
                    #self.logger.debug(f'   PCT_CHG:{pct_chg:.2%} coin:{coin}')
                    df.at[i, 'BUYSELL_PRICE'] = -sell_price
                else:
                    print(f"{sell_price/coin_price - coin:.1f}コイン 不足")

            elif len(position_df) != 0 and self.param.LOGIC != 10 and self.param.SONGIRI == True:
                # 損切り
                # 積み立ての時は実施損切しない
                pass
                df, position_df, coin, yen = self.songiri(df, position_df, coin_price, coin, yen, tmp_df)

            df.at[i, 'SimulateAsset'] = yen + coin*coin_price
            df.at[i, 'Profit'] = df.at[i, 'SimulateAsset'] - init_asset
            df.at[i, 'Coin'] = coin
            df.at[i, 'JPY'] = yen
            df.at[i, 'GachihoAsset'] = init_yen + init_coin*coin_price
            df.at[i, 'GachihoProfit'] = df.at[i, 'GachihoAsset'] - init_asset

            # ポジション保存
            if df.at[i, 'BUYSELL'] == BUY or df.at[i, 'BUYSELL'] == SELL:
                # 売り買いして、
                if df.at[i, 'Songiri'] == False:
                    # それが損切りの売買でない場合
                    bs = df.at[i, 'BUYSELL']
                    t = i.strftime('%Y/%m/%d %H:%M:%S')
                    p = abs(df.at[i, 'BUYSELL_PRICE'])
                    y = df.at[i, 'JPY']
                    c = df.at[i, 'Coin']
                    ast = df.at[i, 'SimulateAsset']
                    pp = 1+df.at[i, 'Profit']/init_asset
                    print(f"[{'買い' if bs == BUY else '売り'}:{t}] 売買価格：{p:.0f} 円：{y:.0f} コイン：{c:.1f} 資産：{ast:.0f} 利益率：{pp:.1%}")
                    position_df = position_df.append(df.loc[i])

            # 所持コイン、所持日本円がマイナスになったら強制終了
            if self.check_minus(df) == -1:
                break

        return df


    def set_ma(self, df):
        """
            移動平均（MA）とRSIを計算、DataFrameに追記し返却
        """
        # 移動平均の差分と、連続GC/DC回数を取得
        df['ma_diff'], df['GCDC_times'] = self.get_madata(df)

        # RSIを取得
        # df['rsi'] = get_rsi(df)

        return df


    def save_gragh(self, df, filename):
        rows = 5
        plt.figure(figsize=(60,10))
        plt.xlim(df.index[0], df.index[-1])

        # 利益グラフ
        plt.subplot(rows,1,1)
        plt.plot(df.index, df["Profit"], label="Profit")
        plt.title("Profit Graph")
        plt.xlabel("Date")
        plt.ylabel("Profit")
        plt.xlim(df.index[0], df.index[-1])

        # ガチホ利益グラフ
        plt.plot(df.index, df["GachihoProfit"], label="GachihoProfit")
        plt.legend()

        # 価格グラフ
        plt.subplot(rows,1,2)
        plt.plot(df.index, df["Close"], label="Close")
        plt.title("Price Graph")
        plt.xlabel("Date")
        plt.ylabel("Price")

        # 売買ポイント
        buydf = df[df['BUYSELL'] == 1]
        selldf = df[df['BUYSELL'] == -1]
        plt.scatter(buydf.index, buydf['Close'], label='BUY', color='red', s=10)
        plt.scatter(selldf.index, selldf['Close'], label='SELL', color='blue', s=10)

        plt.legend()
        plt.xlim(df.index[0], df.index[-1])

        # STOCH FAST
        plt.subplot(rows,1,3)
        plt.plot(df.index, df['STOCHRSI_fastk'], label='STOCHRSI_k')
        plt.plot(df.index, df['STOCHRSI_fastd'], label='STOCHRSI_d')
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("STOCHRSI")
        plt.xlim(df.index[0], df.index[-1])

        # STOCH SLOw
        plt.subplot(rows,1,4)
        plt.plot(df.index, df['STOCH_slowk'], label='STOCH_slowk')
        plt.plot(df.index, df['STOCH_slowd'], label='STOCH_slowd')
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("STOCH SLOW")
        plt.xlim(df.index[0], df.index[-1])

        # 所持コイン
        plt.subplot(rows,1,5)
        plt.plot(df.index, df['Coin'], label='COIN')
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Coin")
        plt.xlim(df.index[0], df.index[-1])

        # plt.subplot(414)
        # plt.plot(df.index, df['RSI'], label='RSI')
        # plt.legend()
        # plt.xlabel("Date")
        # plt.ylabel("RSI")
        # plt.plot(buydf.index, buydf['Close'], marker='o', markersize=5, label='BUY', color='red')
        # plt.plot(selldf.index, selldf['Close'], marker='o', markersize=5, label='SELL', color='blue')


        # for i, r in df.iterrows():
        #     d = pd.DataFrame([r])
        #     nd = pd.DataFrame()
        #
        #     if d['BUYSELL'][0] == BUY:
        #         nd['Date'] = d.index
        #         nd['Close'] = d['Close']
        #         plt.plot(d.index, d['Close'][0], marker='o', markersize=5, label='BUY', color='red')
        #     elif d['BUYSELL'][0] == SELL:
        #         plt.plot(d.index, d['Close'][0], marker='o', markersize=5, label='SELL', color='blue')

        plt.savefig(filename, format="png")

    """
    Force Entry Priceは買うと決めてから約定するまで指値で追いかけた場合に、実際に約定する価格
    force_entry_time: 約定するまでにかかった時間

    やっていること
    1. 毎時刻、与えられた指値価格で、指値を出す
    2. 指値が約定したら、指値をForce Entry Priceとする
    3. 指値が約定しなかったら、次の時刻へ進み、1へ戻る

    """
    def calc_force_entry_price(self, entry_price=None, lo=None, pips=None):
        y = entry_price.copy() # ロングのときは購入価格
        y[:] = np.nan

        # shapeをあわせてNaNで埋める
        force_entry_time = entry_price.copy()
        force_entry_time[:] = np.nan

        """
        時刻i
          ----- i ------------------------------------->
              y[i]

        時刻j
          -------------------- j-1 ---------------- j ------------------>
                    entry_price[j-1]  >  low[j]

        """
        for i in range(entry_price.size):
            for j in range(i + 1, entry_price.size):
              # ある時刻（i）より先の時刻(j)の価格を見る
                if round(lo[j] / pips) < round(entry_price[j - 1] / pips):
                  # 約定した場合（時刻j-1のenrty_price（購入価格）が時刻j安値より高い場合）
                  # ある時刻iの目的変数を約定価格にする
                  y[i] = entry_price[j - 1]
                  # force_entry_time: 約定するまでにかかった時間
                  force_entry_time[i] = j - i
                  if i < 3:
                    print(f"i:{i}, j:{j}, y[{i}]:{y[i]}")
                    print(f"  lo[j] < entry_price[j-1] : {lo[j]} < {entry_price[j-1]}")
                    print(f"  fep[{i}]:{force_entry_time[i]}")
                  break

        return y, force_entry_time

    """
    y計算ロジック
    1. 毎時刻、あるルールで計算された指値距離(limit_price_dist)に基づいて、買い指値を出す
    2. 買い指値が約定しなかった場合のyはゼロとする
    3. 買い指値が約定した場合、一定時間(horizon)だけ待ってから、Force Entry Priceの執行方法でエグジットする
    4. エグジット価格 / エントリー価格 - 1 - 2 * feeをyとする
    """
    def set_y(self, df):
        # 呼び値 (取引所、取引ペアごとに異なるので、適切に設定してください)
        pips = 1

        # ATRで指値距離を計算します
        limit_price_dist = df['ATR'] * 0.5
        limit_price_dist = np.maximum(1, (limit_price_dist / pips).round().fillna(1)) * pips

        # 終値から両側にlimit_price_distだけ離れたところに、買い指値と売り指値を出します
        df['buy_price'] = df['Close'] - limit_price_dist
        df['sell_price'] = df['Close'] + limit_price_dist

        features = sorted([
            'ADX',
            'ADXR',
            'APO',
            'AROON_aroondown',
            'AROON_aroonup',
            'AROONOSC',
            'CCI',
            'DX',
            'MACD_macd',
            'MACD_macdsignal',
            'MACD_macdhist',
            'MFI',
        #     'MINUS_DI',
        #     'MINUS_DM',
            'MOM',
        #     'PLUS_DI',
        #     'PLUS_DM',
            'RSI',
            'STOCH_slowk',
            'STOCH_slowd',
            'STOCHF_fastk',
        #     'STOCHRSI_fastd',
            'ULTOSC',
            'WILLR',
        #     'ADOSC',
        #     'NATR',
            'HT_DCPERIOD',
            'HT_DCPHASE',
            'HT_PHASOR_inphase',
            'HT_PHASOR_quadrature',
            'HT_TRENDMODE',
            'BETA',
            'LINEARREG',
            'LINEARREG_ANGLE',
            'LINEARREG_INTERCEPT',
            'LINEARREG_SLOPE',
            'STDDEV',
            'BBANDS_upperband',
            'BBANDS_middleband',
            'BBANDS_lowerband',
            'DEMA',
            'EMA',
            'HT_TRENDLINE',
            'KAMA',
            'MA',
            'MIDPOINT',
            'T3',
            'TEMA',
            'TRIMA',
            'WMA',
        ])
        df['fee'] = 0.0

        # Force Entry Priceの計算
        # 買いの場合
        print("買い")
        df['buy_fep'], df['buy_fet'] = self.calc_force_entry_price(
            entry_price=df['buy_price'].values,
            lo=df['Low'].values,
            pips=pips,
        )


        # calc_force_entry_priceは入力と出力をマイナスにすれば売りに使えます
        print("売り")
        df['sell_fep'], df['sell_fet'] = self.calc_force_entry_price(
            entry_price=-df['sell_price'].values,
            lo=-df['High'].values, # 売りのときは高値
            pips=pips,
        )
        df['sell_fep'] *= -1


        horizon = 1 # エントリーしてからエグジットを始めるまでの待ち時間 (1以上である必要がある)
        fee = df['fee'] # maker手数料

        # 指値が約定したかどうか (0, 1)
        df['buy_executed'] = ((df['buy_price'] / pips).round() > (df['Low'].shift(-1) / pips).round()).astype('float64')
        df['sell_executed'] = ((df['sell_price'] / pips).round() < (df['High'].shift(-1) / pips).round()).astype('float64')


        # yを計算
        df['y_buy'] = np.where(
            df['buy_executed'], # 約定し、
            df['sell_fep'].shift(-horizon) / df['buy_price'] - 1 - 2 * fee, # horizon時間後に売ったときにどれくらいの割合変化したか？
            0
        )
        df['y_sell'] = np.where(
            df['sell_executed'],
            -(df['buy_fep'].shift(-horizon) / df['sell_price'] - 1) - 2 * fee,
            0
        )

        # バックテストで利用する取引コストを計算
        df['buy_cost'] = np.where(
            df['buy_executed'],
            df['buy_price'] / df['Close'] - 1 + fee,
            0
        )
        df['sell_cost'] = np.where(
            df['sell_executed'],
            -(df['sell_price'] / df['Close'] - 1) + fee,
            0
        )

        # pp(df[['Close', 'Low', 'buy_price', 'buy_fep', 'buy_fet', 'buy_executed', 'y_buy', 'buy_cost', 'sell_price', 'sell_fep', 'sell_fet', 'sell_executed', 'y_sell', 'sell_cost']])

        print('約定確率を可視化。時期によって約定確率が大きく変わると良くない。')
        df['buy_executed'].rolling(1000).mean().plot(label='買い')
        df['sell_executed'].rolling(1000).mean().plot(label='売り')
        plt.title('約定確率の推移')
        plt.legend(bbox_to_anchor=(1.05, 1))
        # plt.show()

        print('エグジットまでの時間分布を可視化。長すぎるとロングしているだけとかショートしているだけになるので良くない。')
        df['buy_fet'].rolling(1000).mean().plot(label='買い')
        df['sell_fet'].rolling(1000).mean().plot(label='売り')
        plt.title('エグジットまでの平均時間推移')
        plt.legend(bbox_to_anchor=(1.2, 1))
        # plt.show()

        df['buy_fet'].hist(alpha=0.3, label='買い')
        df['sell_fet'].hist(alpha=0.3, label='売り')
        plt.title('エグジットまでの時間分布')
        plt.legend(bbox_to_anchor=(1.2, 1))
        # plt.show()

        print('毎時刻、この執行方法でトレードした場合の累積リターン')
        df['y_buy'].cumsum().plot(label='買い')
        df['y_sell'].cumsum().plot(label='売り')
        plt.title('累積リターン')
        plt.legend(bbox_to_anchor=(1.05, 1))
        # plt.show()


    def main(self):
        # オプション引数
        argparser = ArgumentParser()
        argparser.add_argument('-l') # OHLCVファイルの読み込み
        argparser.add_argument('-s', action='store_true', help='Simulate mode.') # シミュレートモード
        argparser.add_argument('--logic') # 売買価格決定ロジックの指定
        argparser.add_argument('-o') # シミュレート結果CSVの出力先指定
        argparser.add_argument('--nosongiri', action='store_false', help='No Songiri mode.') # 損切するかしないか。デフォルトは損切する
        argparser.add_argument('--mlmodel') # 機械学習モデルのファイル名
        argparser.add_argument('--mlinput') # 機械学習するinputファイル名
        argparser.add_argument('--pdl') # price_decision_logic

        args = argparser.parse_args()

        # DataFrameの最大表示行数
        pd.set_option('display.max_rows', 100)

        if args.mlinput is not None:
            """
            機械学習のみ
            """
            input_filename = args.mlinput
            df = pd.read_csv(input_filename)
            df = df.set_index('Date')
            df = df.astype(float)
            df = df.dropna()

            def all_combination(columns):
                result = []
                for n in range(1, len(columns)+1):
                    for conb in itertools.combinations(columns, n):
                        result.append(list(conb)) #タプルをリスト型に変換

                return result

            # ac = all_combination(self.features)
            # features_list = [f for f in ac if len(f) >= 8]
            self.ml = MachineLearning()
            features_list = [self.features]
            print(f"学習回数：{len(features_list)}")

            ml_ret = {}
            # self.ml.show_corr(df)
            # self.ml.reduce(df)

            for f in features_list:
                print(f)
                y = df[['BUYSELL_PRICE']]
                df = df[f] # 未来特徴量は削除

                # 特徴量エンジニアリング
                X = self.ml.feature_engineering(df)
                print(X.columns)

                # 学習実施
                score = self.ml.learn(X, y, 'test_model.pkl')

                ml_ret[",".join(f)] = score

            with open('ml_ret.csv', 'w') as f:
                writer = csv.writer(f)
                for k, v in ml_ret.items():
                   writer.writerow([k, v])

        else:
            """
            売買
            """
            # OHLCVデータ読み込み
            if args.l is not None:
                # CSVファイルからOHLCVデータを読み取り
                df = self.load_csv2pd(args.l)

            else:
                # 前日までのデータを収集
                date = datetime.now()# - timedelta(days=1)
                df = self.get_ohlcv(date, 48, self.param.CANDLE_TYPE)
                # df = self.get_ohlcv(date, 24*365, self.param.CANDLE_TYPE)
                # df.to_csv("./sampledata_365days_ohlcv.csv")
                # sys.exit()

            # 機械学習モデルの読み込み
            if args.mlmodel is not None:
                self.ml = MachineLearning(model_file=args.mlmodel)

            if args.s is True:
                """
                シミュレーションモード
                """
                # 特徴量計算
                df = self.calc_features(df)
                df = self.set_ma(df)
                # NaNを含む行を削除
                df = df.dropna()
                # 対象通貨の現在の価格
                coin_price = df['Close'][-1]

                # 初期パラメータ設定
                init_yen = 100000.0
                init_coin = 100.0
                output_filename = ""

                # コマンドライン引数からパラメータ取得
                if args.logic is not None:
                    self.param.LOGIC = int(args.logic)

                if args.o is not None:
                    output_filename = args.o

                if args.nosongiri is not None:
                    self.param.SONGIRI = args.nosongiri

                if args.pdl is not None:
                    self.param.PDL = int(args.pdl)


                # シミュレーション開始
                sim_df = self.simulate(df,
                    init_yen=init_yen,
                    init_coin=init_coin)

                # 結果（利益）表示
                print(f"シミュレーション利益:{sim_df['Profit'][-1]:.0f}円({1+sim_df['Profit'][-1]/sim_df['SimulateAsset'][0]:.2%})")
                print(f"ガチホ時の利益:{sim_df['GachihoProfit'][-1]:.0f}円({1+sim_df['GachihoProfit'][-1]/sim_df['SimulateAsset'][0]:.2%})")
                print(f"ガチホと比較した効果：{sim_df['SimulateAsset'][-1]/sim_df['GachihoAsset'][-1]:.2%}")

                # シミュレート結果のファイル出力
                if output_filename != "":
                    sim_df.to_csv(output_filename)

                # グラフ描画
                self.save_gragh(sim_df, "simulate00.png")

            else:
                """
                実際の取引
                """

                # 最新のOHLCVデータを取得
                latest_ohlcv = self.get_ohlcv(datetime.now(), 1, self.param.CANDLE_TYPE)
                print(latest_ohlcv.tail(1))
                df = df.append(latest_ohlcv.tail(1))
                # 特徴量計算
                df = self.calc_features(df)
                df = df.drop(['_CLOSE_PCT_CHANGE'], axis=1) # 未来特徴量は削除
                df = self.set_ma(df)
                # print(df)
                # df.to_csv('test.csv')
                df = df.dropna()
                # print(df)

                if self.buyORsell(df) == SELL:
                    # ##################
                    # 売　却
                    # ##################
                    buy_price = self.get_BUYSELLprice(self.param.BUY_PRICE, coin_price, coin, yen)  # 購入する仮想通貨の枚数

                    if yen > buy_price:
                        df['BUYSELL'][-1] = BUY
                        yen -= (buy_price + self.fee(buy_price))
                        coin += buy_price/coin_price
                        df['BUYSELL_PRICE'][-1] = buy_price
                    else:
                        print(f"{buy_price-yen:.0f}円 不足")

                    self.order(SELL, buy_price, coin_price)

                elif self.buyORsell(df) == BUY:
                    # ##################
                    # 購　入
                    # ##################
                    sell_price = self.get_BUYSELLprice(self.param.SELL_PRICE, coin_price, coin, yen)  # 購入する仮想通貨の枚数

                    if coin > sell_price/coin_price:
                        df['BUYSELL'][-1] = SELL
                        yen += (sell_price - self.fee(sell_price))
                        coin -= sell_price/coin_price
                        df['BUYSELL_PRICE'][-1] = -sell_price
                    else:
                        print(f"{sell_price/coin_price - coin:.1f}コイン 不足")

                    self.order(BUY, BUY_PRICE, coin_price)


if __name__ == "__main__":
    param = Parameter(buy_price=100, sell_price=100)
    at = AutoTrade(param)

    at.main()
