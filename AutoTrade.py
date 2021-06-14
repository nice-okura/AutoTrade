import base64
import hmac
import hashlib
import json
import sys
import numpy as np
import math
import urllib
import time
from datetime import datetime
from datetime import timedelta
import os
import csv
import pandas as pd
import logging

import requests
from pprint import pprint as pp

DEBUG = True

# 環境変数API_KEY, API_SECRETの設定が必要
#
# $ API_KEY='xxxx'
# $ API_SECRET='xxxx'
API_KEY=os.environ['API_KEY']
API_SECRET=os.environ['API_SECRET']
CURRENCY_LIST = ['btc', 'xrp', 'xem', 'eth', 'jpy', 'ltc', 'bcc', 'mona', 'bat', 'qtum']
URL = "https://api.bitbank.cc"
PUBLIC_URL = "https://public.bitbank.cc"
PRIVATE_URL = "https://api.bitbank.cc/v1"

# Config
MA_short = 5 # 移動平均（短期）
MA_long = 50 # 移動平均（長期）
TIMEOUT = 5
CANDLE_TYPE = '1hour' # データ取得間隔
PAIR = 'qtum_jpy' # 対象通貨
MA_times = 12 # コインを購入/売却金額する連続GC/DC回数
BUY_PRICE = 500.0 # 購入金額(円)
SELL_PRICE = 500.0 # 売却金額(円)

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


def createSign(pParams, method, host_url, request_path, secret_key):
    encode_params = json.dumps(pParams)
    payload = request_path+encode_params
    payload = payload.encode(encoding='UTF8')
    secret_key = secret_key.encode(encoding='UTF8')
    digest = hmac.new(secret_key, payload, digestmod=hashlib.sha256).hexdigest()

    return digest

def http_get_request(url, params, add_to_headers=None):
    headers = {
    }
    if add_to_headers:
        headers.update(add_to_headers)
    postdata = urllib.parse.urlencode(params)
    # print(headers)
    try:
        response = requests.get(url, postdata, headers=headers, timeout=TIMEOUT)
        # pprint(url)
        if response.status_code == 200:
            return response.json()
        else:
            return response.json()
            # return {"status": "fail"}
    except Exception as e:
        print("httpGet failed, detail is:%s" %e)
        return {"status":"fail","msg": "%s"%e}

def http_post_request(url, params, add_to_headers=None):
    headers = {
        'Content-Type': 'application/json'
    }
    if add_to_headers:
        headers.update(add_to_headers)
    postdata = json.dumps(params)

    try:
        response = requests.post(url, postdata, headers=headers, timeout=TIMEOUT)
        if response.status_code == 200:
            return response.json()
        else:
            return response.json()
    except Exception as e:
        print("httpPost failed, detail is:%s" % e)
        return {"status":"fail","msg": "%s"%e}

def api_key_get(url, request_path, params, ACCESS_KEY, SECRET_KEY):
    # UNIX時間
    utctime = str(int(time.time()))

    params = {
        'ACCESS-NONCE':utctime,
        'ACCESS-SIGNATURE':createSign([], "GET", "", utctime+request_path, SECRET_KEY),
        'ACCESS-KEY':ACCESS_KEY
        }
    # pprint(params)
    return http_get_request(url+request_path, [], params)

def api_key_post(url, request_path, params, access_key, secret_key):
    method = 'POST'
    # UNIX時間
    utctime = str(int(time.time()))

    headers = {
        'ACCESS-NONCE':utctime,
        'ACCESS-SIGNATURE':createSign(params, "", "", utctime, secret_key),
        'ACCESS-KEY':access_key
        }

    return http_post_request(url+request_path, params, headers)

def get_assets():
    request_path = "/v1/user/assets"

    return api_key_get(URL, request_path, [], API_KEY, API_SECRET)

def get_ticker(pair):
    request_path = "/"+ pair + "/ticker"

    return api_key_get(PUBLIC_URL, request_path, [], API_KEY, API_SECRET)

def get_candlestick(candletype, pair, day):
    request_path = "/" + pair + "/candlestick/" + candletype + "/" + day
    # pp(PUBLIC_URL+request_path)
    return api_key_get(PUBLIC_URL, request_path, [], "", "")

def daterange(_start, _end):
    for n in range((_end - _start).days):
        yield _start + timedelta(n)

def post_order(pair, amount, price, side, type, post_only=False):
    request_path = "/user/spot/order"
    params = {
        'pair': pair,
        'amount': amount,
        'side': side,
        'type': type,
    }
    if price != None:
        params["price"] = price
    if post_only != None:
        params["post_only"] = post_only

    return api_key_post(PRIVATE_URL, request_path, params, API_KEY, API_SECRET)

def get_ohlcv(date, size):
    df = pd.DataFrame()

    # 必要なチャートを取得
    while(len(df) < size):
        d_str = date.strftime('%Y%m%d')
        logger.debug(d_str + "分 データ取得開始")
        cd = get_candlestick(CANDLE_TYPE, PAIR, d_str)

        if cd['success'] == 0:
            logger.info(d_str + "分 データなし")
            logger.info(cd)
            date = date - timedelta(days=1)
            continue

        ohlcv = get_candlestick(CANDLE_TYPE, PAIR, d_str)["data"]["candlestick"][0]["ohlcv"]
        columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']

        df_new = pd.DataFrame(data=ohlcv, columns=columns)
        df_new['Date']=pd.to_datetime(df_new['Date'], unit='ms', utc=True)
        df_new = df_new.set_index('Date')
        df_new = df_new.astype(float)
        df_new.index = df_new.index.tz_convert('Asia/Tokyo')

        df = pd.concat([df_new, df])
        date = date - timedelta(days=1)

    return df

if __name__ == "__main__":

    date = datetime.now()
    pd.set_option('display.max_rows', 10)

    df = get_ohlcv(date, MA_long*2)

    # 移動平均の差(ma_diff)を計算
    ma_short = df.rolling(MA_short).mean()
    ma_long = df.rolling(MA_long).mean()
    df['ma_short'] = ma_short['Close']
    df['ma_long'] = ma_long['Close']
    df['ma_diff'] = ma_short['Close'] - ma_long['Close']

    # 連続GC/DC回数を計算
    times_list = []
    for i, d in enumerate(df['ma_diff']):
        # 以下のいずれか場合は連続GC/DC回数を初期値（1）に設定
        # 　初めのデータ：i == 0
        #　 差分データがないとき：math.isnan(d)
        # 　前回データがないとき：math.isnan(df['ma_diff'][i-1])
        # 　GC -> DCまたはDC -> GCのとき：d * df['ma_diff'][i-1] < 0
        if i == 0 or math.isnan(d) or math.isnan(df['ma_diff'][i-1]) or d * df['ma_diff'][i-1] < 0 :
            times_list.append(1)
        elif d is not np.nan:
            times_list.append(times_list[i-1]+1)
    df = df.assign(GCDC_times=times_list)

    logger.info("\n" + str(df.tail(10)))

    # MA_times回連続してGC/DCした場合、コインを購入/売却する
    if df['GCDC_times'][-1] % MA_times == 0:

        coin_price = df['Close'][-1]
        logger.debug("df['GCDC_times'][-1]: " + str(df['GCDC_times'][-1]) + " coin_price: " + str(coin_price))

        # GCなので売却
        if df['ma_diff'][-1] < 0:
            price = SELL_PRICE/coin_price
            if DEBUG == False:
                order_result = post_order(PAIR, price, "", "sell", "market")
            else:
                order_result = {'success': 0, 'data': "デバッグモード"}
            if order_result['success'] == 1:
                # オーダー成功
                price = order_result['data']['start_amount']
                logger.info(PAIR + "を" + str(SELL_PRICE) + "円で" + str(price) + " 売却")
            else:
                # オーダー失敗
                logger.error("オーダー失敗")
                logger.error(order_result)
        if df['ma_diff'][-1] >= 0:
            price = BUY_PRICE/coin_price
            if DEBUG == False:
                order_result = post_order(PAIR, price, "", "buy", "market")
            else:
                order_result = {'success': 0, 'data': "デバッグモード"}
            if order_result['success'] == 1:
                # オーダー成功
                price = order_result['data']['start_amount']
                logger.info(PAIR + "を" + str(BUY_PRICE) + "円で" + str(price) + " 購入")
            else:
                # オーダー失敗
                logger.error("オーダー失敗")
                logger.error(order_result)
