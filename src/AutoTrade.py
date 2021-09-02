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
from CryptService import CryptService

from pprint import pprint as pp

DEBUG = True

# 0: 通常モード
# 1: 過去のデータから売り買い判定のみ表示
# 2: 売買シミュレーション結果を表示
MODE = 2

URL = "https://api.bitbank.cc/v1"
PUBLIC_URL = "https://public.bitbank.cc"

# Config
MA_short = 5 # 移動平均（短期）
MA_long = 50 # 移動平均（長期）
CANDLE_TYPE = '1hour' # データ取得間隔
PAIR = 'qtum_jpy' # 対象通貨
MA_times = 6 # コインを購入/売却金額する連続GC/DC回数
BUY_PRICE = 500.0 # 購入金額(円)
SELL_PRICE = 500.0 # 売却金額(円)
RSI_SELL = 70.0 # 売りRSIボーダー
RSI_BUY = 100.0 - RSI_SELL # 買いRSIボーダー
VOL_ORDER = 20000 # 取引する基準となる取引量(Volume)
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

cs = CryptService(URL, PUBLIC_URL, os.environ['API_KEY'], os.environ['API_SECRET'], "bitbank")

def get_ohlcv(date, size, candle_type):
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
        logger.debug(d_str + "分 データ取得開始")
        cd = cs.get_candlestick(candle_type, PAIR, d_str)

        if cd['success'] == 0:
            logger.info(d_str + "分 データなし")
            logger.info(cd)

            date = date - timedelta(days=1)
            continue

        ohlcv = cs.get_candlestick(candle_type, PAIR, d_str)["data"]["candlestick"][0]["ohlcv"]
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
        order_result = cs.post_order(PAIR, price, "", order_mode, "market")
    else:
        order_result = {'success': 0, 'data': "デバッグモード"}
    if order_result['success'] == 1:
        # オーダー成功
        price = order_result['data']['start_amount']
        logger.info(PAIR + "を" + str(price_yen) + "円で" + str(price) + order_str)

        return price
    else:
        # オーダー失敗
        logger.error("オーダー失敗")
        logger.error(order_result)

        return -1

    return -1

def buyORsell(df, logic=0):
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

    if 'ma_diff' in df.columns:
        gcdc = "GC" if df['ma_diff'][-1] >= 0 else "DC"
    else:
        logger.warning("ma_diff カラムがありません")
        return buysell

    """
        メインの売り買いロジック
    """
    if logic == 0:
        if buysell_by_vol(df):
            if (is_gcdc(df, MA_times) and gcdc == "GC") or buysell_by_rsi(df) == BUY:
                buysell = BUY
            elif (is_gcdc(df, MA_times) and gcdc == "DC") or buysell_by_rsi(df) == SELL:
                buysell = SELL
    else:
        logger.error("対応ロジックなし logic: " + logic)

    return buysell

def show_buysellpoint(df):
    """
        過去のデータ(df)から、売り買いポイントをすべて表示
    """
    df['BUYSELL'] = 0

    for i in range(len(df)):
        tmp_df = df.iloc[i:i+1]
        if buyORsell(tmp_df) == BUY:
            df.iat[i,8] = "BUY"
        elif buyORsell(tmp_df) == SELL:
            df.iat[i,8] = "SELL"
    logger.info("\n" + str(df.tail(100)))

def simulate(df):
    """
        過去データ(df)から実際に売買した場合の総資産や利益を表示
    """
    yen = 100000 # 初期日本円
    coin = 100 # 初期仮想通貨数
    init_asset  = 100000 + 100*df['Close'][0]
    df['BUYSELL'] = 0             # 売り買いの識別　index 8
    df['SimulateAsset'] = 0.0     # シミュレーションしたときの総資産　index 9
    df['Profit'] = 0.0            # シミュレーションしたときの利益（総資産ー初期資産）index 10
    df['Coin'] = 0.0              # 所持仮想通貨数　index 11

    for i in range(len(df)):
        tmp_df = df.iloc[i:i+1]
        coin_price = tmp_df['Close'][0]

        if buyORsell(tmp_df) == BUY:
            df.iat[i,8] = "BUY"
            price = BUY_PRICE/coin_price
            yen -= BUY_PRICE
            coin += price

        elif buyORsell(tmp_df) == SELL:
            df.iat[i,8] = "SELL"
            price = SELL_PRICE/coin_price
            yen += SELL_PRICE
            coin -= price

        df.iat[i,9] = yen + coin*coin_price # SimulateAsset
        df.iat[i,10] = df.iat[i,9] - init_asset # Profit
        df.iat[i,11] = coin # Coin

    # シミュレーション結果を表示
    logger.info("\n" + str(df.tail(100)))

if __name__ == "__main__":
    pd.set_option('display.max_rows', 100)

    date = datetime.now() - timedelta(days=1)
    df = get_ohlcv(date, MA_long*2, CANDLE_TYPE)

    # 移動平均の差分と、連続GC/DC回数を取得
    df['ma_diff'], df['GCDC_times'] = get_madata(df)

    # RSIを取得
    df['rsi'] = get_rsi(df)

    logger.info("\n" + str(df.tail(40)))

    # 対象通貨の現在の価格
    coin_price = df['Close'][-1]

    if MODE == 1:
        show_buysellpoint(df)
    elif MODE == 2:
        simulate(df)
    elif MODE == 3:
        if buyORsell(df) == SELL:
            # ##################
            # 売　却
            # ##################
            order(SELL, SELL_PRICE, coin_price)

        elif buyORsell(df) == BUY:
            # ##################
            # 購　入
            # ##################
            order(BUY, BUY_PRICE, coin_price)
