import json
import sys
import numpy as np
import math
import time
from datetime import datetime
from datetime import timedelta
import os
import csv
import pandas as pd
import logging
from OrderUtils import *

from pprint import pprint as pp

DEBUG = True

# Config
MA_short = 5 # 移動平均（短期）
MA_long = 100 # 移動平均（長期）
CANDLE_TYPE = '1hour' # データ取得間隔
PAIR = 'qtum_jpy' # 対象通貨
MA_times = 1 # コインを購入/売却金額する連続GC/DC回数
BUY_PRICE = 1.0 # 購入金額(円)
SELL_PRICE = 1.0 # 売却金額(円)
RSI_SELL = 50.0 # 売りRSIボーダー
RSI_BUY = 100.0 - RSI_SELL # 買いRSIボーダー
VOL_ORDER = 5000 # 取引する基準となる取引量(Volume)
BUY = 1
SELL = -1

# 1. ロガーを取得する
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # 出力レベルを設定

# 2. ハンドラーを生成する
h = logging.StreamHandler()
h.setLevel(logging.DEBUG) # 出力レベルを設定
h2 = logging.FileHandler('./AutoTrade.log')
if DEBUG == True:
    h2.setLevel(logging.DEBUG) # 出力レベルを設定
else:
    h2.setLevel(logging.INFO) # 出力レベルを設定

# 3. フォーマッタを生成する
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 4. ハンドラーにフォーマッターを設定する
h.setFormatter(fmt)
h2.setFormatter(fmt)

# 5. ロガーにハンドラーを設定する
logger.addHandler(h)
logger.addHandler(h2)

# dateからsize単位時間分のOHLCVデータを取得する
def get_ohlcv(date, size, candle_type):
    ohlcv_df = pd.DataFrame()

    # 必要なチャートを取得
    while(len(ohlcv_df) < size):
        d_str = date.strftime('%Y%m%d')
        logger.debug(d_str + "分 データ取得開始")
        cd = get_candlestick(candle_type, PAIR, d_str)

        if cd['success'] == 0:
            logger.info(d_str + "分 データなし")
            logger.info(cd)

            date = date - timedelta(days=1)
            continue

        ohlcv = get_candlestick(candle_type, PAIR, d_str)["data"]["candlestick"][0]["ohlcv"]
        columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']

        df_new = pd.DataFrame(data=ohlcv, columns=columns)
        df_new['Date']=pd.to_datetime(df_new['Date'], unit='ms', utc=True)
        df_new = df_new.set_index('Date')
        df_new = df_new.astype(float)
        df_new.index = df_new.index.tz_convert('Asia/Tokyo')

        ohlcv_df = pd.concat([df_new, ohlcv_df])
        date = date - timedelta(days=1)

    return ohlcv_df

# 移動平均の差(ma_diff)と連続GC/DC回数を計算し、返却
def get_madata(df):
    ma_short = df.rolling(MA_short).mean()
    ma_long = df.rolling(MA_long).mean()
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

# RSIを計算
def get_rsi(df):
    # RSI計算
    diff = df['Close'] - df['Open']
    up, down = diff.copy(), diff.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    up_sma_14 = up.rolling(window=14, center=False).mean()
    down_sma_14 = down.abs().rolling(window=14, center=False).mean()
    rs = up_sma_14 / down_sma_14
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi

# 連続times x n回 DCまたはGCを継続しているか判定
def is_gcdc(df, times):
    return df['GCDC_times'][-1] % times == 0

# RSIをもとに売り(-1) or 買い(1) or ステイ(0)を判定
def buysell_by_rsi(df):
    buysell = 0

    if df['rsi'][-1] <= RSI_BUY:
        buysell = BUY
    elif df['rsi'][-1] >= RSI_SELL:
        buysell = SELL

    return buysell

# Volumeをもとに取引するかしないを判定
def buysell_by_vol(df):
    return df['Volume'][-1] >= VOL_ORDER

def order(buysell, price_yen, coin_price):
    """
    コインを購入もしくは売却
    """
    price = price_yen/coin_price
    order_mode, order_str = ("buy", "購入") if buysell == BUY else ("sell", "売却")

    if DEBUG == False:
        order_result = post_order(PAIR, price, "", order_mode, "market")
    else:
        order_result = {'success': 0, 'data': "デバッグモード"}
    if order_result['success'] == 1:
        # オーダー成功
        price = order_result['data']['start_amount']
        logger.info(PAIR + "を" + str(price_yen) + "円で" + str(price) + order_str)
    else:
        # オーダー失敗
        logger.error("オーダー失敗")
        logger.error(order_result)

if __name__ == "__main__":
    pd.set_option('display.max_rows', 30)

    date = datetime.now() - timedelta(days=1)
    df = get_ohlcv(date, MA_long*2, CANDLE_TYPE)

    # 移動平均の差分と、連続GC/DC回数を取得
    df['ma_diff'], df['GCDC_times'] = get_madata(df)

    # RSIを取得
    df['rsi'] = get_rsi(df)

    logger.info("\n" + str(df.tail(10)))

    # MA_times回連続してGC/DCした場合、コインを購入/売却する
    if is_gcdc(df, MA_times) and buysell_by_vol(df):
        gcdc = "GC" if df['ma_diff'][-1] >= 0 else "DC"

        coin_price = df['Close'][-1]
        logger.debug("df['GCDC_times'][-1]: " + str(df['GCDC_times'][-1]) + " coin_price: " + str(coin_price))

        # ##################
        # 売　却
        # ##################
        if gcdc == "DC" or buysell_by_rsi(df) == SELL:
            order(SELL, SELL_PRICE, coin_price)

        # ##################
        # 購　入
        # ##################
        if gcdc == "GC" or buysell_by_rsi(df) == BUY:
            order(BUY, BUY_PRICE, coin_price)
