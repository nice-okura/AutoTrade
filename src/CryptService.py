from CryptUtils import api_key_get, api_key_post

class CryptService:

    def __init__(self, url, public_url, access_key, secret_key):
        self.__url = url
        self.__public_url = public_url
        self.__access_key = access_key
        self.__secret_key = secret_key

    def get_assets(self):
        request_path = "/user/assets"

        return api_key_get(self.__url, request_path, [], self.__access_key, self.__secret_key)

    def get_ticker(self, pair):
        request_path = "/"+ pair + "/ticker"

        return api_key_get(self.__public_url, request_path, [], self.__access_key, self.__secret_key)

    def get_candlestick(self, candletype, pair, day):
        request_path = "/" + pair + "/candlestick/" + candletype + "/" + day

        return api_key_get(self.__public_url, request_path, [], "", "")

    def post_order(self, pair, amount, price, side, type, post_only=False):
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

        return api_key_post(self.__url, request_path, params, self.__access_key, self.__secret_key)
