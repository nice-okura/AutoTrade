import numpy as np
import math
from datetime import datetime
from datetime import timedelta
import os
import pandas as pd
import logging
from CryptService import CryptService
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pprint import pprint as pp
import talib

DEBUG = True

URL = "https://api.bitbank.cc/v1"
PUBLIC_URL = "https://public.bitbank.cc"

# Config
MA_short = 5  # 移動平均（短期）
MA_long = 50  # 移動平均（長期）
CANDLE_TYPE = '1hour'  # データ取得間隔
PAIR = 'qtum_jpy'  # 対象通貨
MA_times = 12  # コインを購入/売却金額する連続GC/DC回数
BUY_PRICE = 400.0  # 購入金額(円)
SELL_PRICE = 400.0  # 売却金額(円)
RSI_SELL = 65.0  # 売りRSIボーダー
RSI_BUY = 100.0 - RSI_SELL  # 買いRSIボーダー
VOL_ORDER = 20  # 取引する基準となる取引量(Volume)
BUY = 1
SELL = -1
WEIGHT_OF_PRICE = 0.05  # 連続MA回数から購入金額を決めるときの重み

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

cs = CryptService(URL, PUBLIC_URL, os.environ['API_KEY'], os.environ['API_SECRET'], "bitbank")

def calc_features(df):
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
    # df['DEMA'] = talib.DEMA(close, timeperiod=30) - hilo
    # df['EMA'] = talib.EMA(close, timeperiod=30) - hilo
    # df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close) - hilo
    # df['KAMA'] = talib.KAMA(close, timeperiod=30) - hilo
    df['MA_SHORT'] = talib.MA(close, timeperiod=MA_short, matype=0) - hilo
    df['MA_LONG'] = talib.MA(close, timeperiod=MA_long, matype=0) - hilo
    df['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
    # df['SMA'] = talib.SMA(close, timeperiod=30) - hilo
    # df['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
    # df['TEMA'] = talib.TEMA(close, timeperiod=30) - hilo
    # df['TRIMA'] = talib.TRIMA(close, timeperiod=30) - hilo
    # df['WMA'] = talib.WMA(close, timeperiod=30) - hilo

    # df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    # df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
    # df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    # df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(high, low, timeperiod=14)
    # df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    # df['BOP'] = talib.BOP(open, high, low, close)
    # df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    # df['DX'] = talib.DX(high, low, close, timeperiod=14)
    df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # skip MACDEXT MACDFIX たぶん同じなので
    # df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    # df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    # df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
    # df['MOM'] = talib.MOM(close, timeperiod=10)
    # df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    # df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
    df['RSI'] = talib.RSI(close, timeperiod=14)
    # df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    # df['TRIX'] = talib.TRIX(close, timeperiod=30)
    # df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    # df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

    # df['AD'] = talib.AD(high, low, close, volume)
    # df['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['OBV'] = talib.OBV(close, volume)

    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    # df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
    # df['TRANGE'] = talib.TRANGE(high, low, close)
    #
    # df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    # df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    # df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
    # df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close)
    # df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
    #
    # df['BETA'] = talib.BETA(high, low, timeperiod=5)
    # df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
    # df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14) - close
    # df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    # df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
    # df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    df['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)
    df['_CLOSE_PCT_CHANGE'] = -close.pct_change(-5) # ５単位時間後との価格差

    return df

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


def load_csv2pd(filename):
    """CSVファイルからDataFrameを読み込み、返す

    """
    df = pd.read_csv(filename, parse_dates=[0])
    df = df.set_index('Date')
    df = df.astype(float)

    return df

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

    if df['RSI'][-1] <= RSI_BUY:
        buysell = BUY
    elif df['RSI'][-1] >= RSI_SELL:
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

    if DEBUG is False:
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
    elif logic == 1:
        """
        売り買いロジック③：
        　MAのみで判断。連続n回GCなら「買い」、連続n回DCなら「売り」
        """
        if is_gcdc(df, MA_times) and gcdc == "GC":
            buysell = BUY
        elif is_gcdc(df, MA_times) and gcdc == "DC":
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
        if buysell_by_vol(df):
            if buysell_by_rsi(df) == BUY:
                buysell = BUY
            elif buysell_by_rsi == SELL:
                buysell = SELL
            else:
                if is_gcdc(df, MA_times) and gcdc == "GC":
                    buysell = BUY
                elif is_gcdc(df, MA_times) and gcdc == "DC":
                    buysell = SELL
    elif logic == 3:
        """
        RSIだけで判断
        """
        if buysell_by_rsi(df) == BUY:
            buysell = BUY
        elif buysell_by_rsi == SELL:
            buysell = SELL

    elif logic == -1:
        """
        （注意：未来データを使ったテスト指標なので実際には使えない）
        ５単位時間後にどのくらい価格変化するかを表した指標（_CLOSE_PCT_CHANGE）が
        border%以上変化する場合、売買する。
        """
        border = 0.03

        if df['_CLOSE_PCT_CHANGE'][-1] >= border:
            buysell = BUY
        elif df['_CLOSE_PCT_CHANGE'][-1] <= -border:
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
            df.iat[i, 8] = "BUY"
        elif buyORsell(tmp_df) == SELL:
            df.iat[i, 8] = "SELL"
    logger.info("\n" + str(df.tail(100)))


def get_BUYSELLprice(yen_price, coin_price, coin, jpy, price_decision_logic=0, oneline_df=None):
    BUYSELLprice = 0.0

    if price_decision_logic == 0:
        BUYSELLprice = yen_price

    elif price_decision_logic == 1 and oneline_df is not None:
        """
        連続GC/DC回数と、WEIGHT_OF_PRICE(重み付け)から売り買い価格を決める
        """
        BUYSELLprice = yen_price * oneline_df['GCDC_times'][0] * WEIGHT_OF_PRICE
        # BUYSELLprice = yen_price * np.log10(oneline_df['GCDC_times'][0])
    elif price_decision_logic == 2 and oneline_df is not None:
        """
        連続GC/DC回数と、WEIGHT_OF_PRICE(重み付け)から売り買い価格を決める
        売り価格は所持仮想通貨数(coin)、買い価格は所持日本円(jpy)から決める
        """
        gcdt_times = oneline_df['GCDC_times'][0]
        weight = (gcdt_times/MA_times * WEIGHT_OF_PRICE)*0.3
        # logger.debug(f"{weight=:.2%}")
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
            BUYSELLprice = coin*coin_price*np.abs(pct_chg*2.0)
        elif pct_chg > 0:
            # 買い
            BUYSELLprice = jpy*np.abs(pct_chg*2.0)

        # BUYSELLprice = yen_price * (1.0+np.abs(oneline_df['_CLOSE_PCT_CHANGE'][0]))*4

    return BUYSELLprice


def simulate(df, logic=0, init_yen=100000, init_coin=100, price_decision_logic=0):
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

    for i, r in df.iterrows():
        tmp_df = pd.DataFrame([r])

        coin_price = tmp_df['Close'][0]  # 購入する仮想通貨の現在の価格

        pct_chg = tmp_df['_CLOSE_PCT_CHANGE'][0]

        if buyORsell(tmp_df, logic) == BUY:
            df.at[i, 'BUYSELL'] = BUY

            buy_price = get_BUYSELLprice(BUY_PRICE, coin_price, coin, yen, price_decision_logic=price_decision_logic, oneline_df=tmp_df)  # 購入する仮想通貨の枚数
            yen -= buy_price
            coin += buy_price/coin_price
            # logger.debug(f'[BUY]{tmp_df.index.strftime("%Y/%m/%d %H:%M")[0]}: BUY_PRICE: {buy_price:.2f} {coin=:.2f}')
            # logger.debug(f'   PCT_CHG:{pct_chg:.2%} jpy:{yen}')

        elif buyORsell(tmp_df, logic) == SELL:
            df.at[i, 'BUYSELL'] = SELL
            sell_price = get_BUYSELLprice(SELL_PRICE, coin_price, coin, yen, price_decision_logic=price_decision_logic, oneline_df=tmp_df)  # 購入する仮想通貨の枚数
            yen += sell_price
            coin -= sell_price/coin_price
            # logger.debug(f'[SELL]{tmp_df.index.strftime("%Y/%m/%d %H:%M")[0]}: SELL_PRICE: {sell_price:.2f} {coin=:.2f}')
            # logger.debug(f'   PCT_CHG:{pct_chg:.2%} coin:{coin}')

        df.at[i, 'SimulateAsset'] = yen + coin*coin_price
        df.at[i, 'Profit'] = df.at[i, 'SimulateAsset'] - init_asset
        df.at[i, 'Coin'] = coin
        df.at[i, 'JPY'] = yen
        df.at[i, 'GachihoAsset'] = init_yen + init_coin*coin_price
        df.at[i, 'GachihoProfit'] = df.at[i, 'GachihoAsset'] - init_asset

    return df


def set_ma_rsi(df):
    """
        移動平均（MA）とRSIを計算、DataFrameに追記し返却
    """
    # 移動平均の差分と、連続GC/DC回数を取得
    df['ma_diff'], df['GCDC_times'] = get_madata(df)

    # RSIを取得
    df['rsi'] = get_rsi(df)

    return df

def save_gragh(df, filename):
    plt.figure(figsize=(60,10))
    plt.subplot(411)
    plt.plot(df.index, df["Profit"], label="Profit")
    plt.title("Profit Graph")
    plt.xlabel("Date")
    plt.ylabel("Profit")

    plt.subplot(412)
    plt.plot(df.index, df["Close"], label="Close")
    plt.title("Price Graph")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.subplot(411)
    plt.plot(df.index, df["GachihoProfit"], label="GachihoProfit")
    plt.legend()

    plt.subplot(412)
    buydf = df[df['BUYSELL'] == 1]
    selldf = df[df['BUYSELL'] == -1]
    plt.scatter(buydf.index, buydf['Close'], label='BUY', color='red')
    plt.scatter(selldf.index, selldf['Close'], label='SELL', color='blue')
    plt.legend()

    plt.subplot(413)
    plt.plot(df.index, df['STOCHRSI_fastk'], label='STOCHRSI_k')
    plt.plot(df.index, df['STOCHRSI_fastd'], label='STOCHRSI_d')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("STOCHRSI")

    plt.subplot(414)
    plt.plot(df.index, df['RSI'], label='RSI')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("RSI")
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
def calc_force_entry_price(entry_price=None, lo=None, pips=None):
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
def set_y(df):
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
    df['buy_fep'], df['buy_fet'] = calc_force_entry_price(
        entry_price=df['buy_price'].values,
        lo=df['Low'].values,
        pips=pips,
    )


    # calc_force_entry_priceは入力と出力をマイナスにすれば売りに使えます
    print("売り")
    df['sell_fep'], df['sell_fet'] = calc_force_entry_price(
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

def set_parameter(ma_short=MA_short, ma_long=MA_long, ma_times=MA_times, vol_order=VOL_ORDER):
    global MA_short
    global MA_long
    global MA_times
    global VOL_ORDER

    MA_short = ma_short
    MA_long = ma_long
    MA_times = ma_times
    VOL_ORDER = vol_order

def main():
    # オプション引数
    argparser = ArgumentParser()
    argparser.add_argument('-l')
    argparser.add_argument('-s', action='store_true', help='Simulate mode.')
    argparser.add_argument('--logic')
    args = argparser.parse_args()

    # DataFrameの最大表示行数
    pd.set_option('display.max_rows', 100)

    if args.l is not None:
        # CSVファイルからOHLCVデータを読み取り
        df = load_csv2pd(args.l)
        df = calc_features(df)

    else:
        # 前日までのデータを収集
        date = datetime.now() - timedelta(days=1)
        df = get_ohlcv(date, MA_long*2, CANDLE_TYPE)
        # df = get_ohlcv(date, 24*200, CANDLE_TYPE)

    # 移動平均(MA)とRSIを計算、設定
    df = set_ma_rsi(df)
    # df.to_csv("sampledata_200days.csv")
    # logger.info("\n" + str(df.tail(40)))

    # 対象通貨の現在の価格
    coin_price = df['Close'][-1]

    # # 目標変数を設定
    # set_y(df)

    if args.s is not None:
        # シミュレーション
        # set_parameter(ma_times=ma_times)

        # 初期パラメータ設定
        logic = 2
        init_yen = 100000
        init_coin = 100
        price_decision_logic = -1

        # コマンドライン引数からパラメータ取得
        if args.logic is not None:
            logic = int(args.logic)

        # シミュレーション開始
        sim_df = simulate(df,
            logic=logic,
            init_yen=init_yen,
            init_coin=init_coin,
            price_decision_logic=price_decision_logic)

        # 結果（利益）表示
        print(f"シミュレーション利益:{sim_df['Profit'][-1]:.0f}円({1+sim_df['Profit'][-1]/sim_df['SimulateAsset'][0]:.2%})")

        # はじめからガチホしていた場合
        print(f"ガチホ時の利益:{sim_df['GachihoProfit'][-1]:.0f}円({1+sim_df['GachihoProfit'][-1]/sim_df['SimulateAsset'][0]:.2%})")

        # 利益 / ガチホ利益
        print(f"ガチホと比較した効果：{sim_df['SimulateAsset'][-1]/sim_df['GachihoAsset'][-1]:.2%}")
        # sim_df.to_csv("sampledata_30days_result.csv")
        save_gragh(sim_df, "simulate00.png")

        # df内の各パラメータの相関を確認
        # import seaborn as sns
        # df_corr = df.corr()
        # plt.figure(figsize=(60,40))
        # sns.heatmap(df_corr, annot=True)
        # plt.title("Corr Heatmap")
        # plt.savefig("heatmap.png", format="png")

    else:
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

if __name__ == "__main__":
    main()
