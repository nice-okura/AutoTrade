import requests
import time
import json
import base64
import hmac
import hashlib
import urllib
import os
from pprint import pprint as pp

TIMEOUT = 5

# 環境変数API_KEY, API_SECRETの設定が必要
#
# $ API_KEY='xxxx'
# $ API_SECRET='xxxx'
# API_KEY=os.environ['API_KEY']
# API_SECRET=os.environ['API_SECRET']

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
    response = http_get_request(url+request_path, [], params)

    if response['success'] == 0:
        if response['data']['code'] == 10000:
            # 10000 URLが存在しません
            raise Exception("URLが存在しない、もしくはまだデータが作成されていない可能性があります。[" + url+request_path + "]")
    return response

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
