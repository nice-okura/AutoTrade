from CryptUtils import api_key_get, api_key_post, setup_csvrowdata, create_data_and_fieldnames, write_csvfile

class CryptService:
    def __init__(self, url, public_url, access_key, secret_key, exchange_name):
        self.__url = url
        self.__public_url = public_url
        self.__access_key = access_key
        self.__secret_key = secret_key
        self.__exchange_name = exchange_name

    def get_assets(self):
        request_path = "/v1/user/assets"

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

    def create_currency_asset(self):
        balance_list, total_assets, currency_list = self.get_currency_data()

        data, dt_month = setup_csvrowdata(total_assets)
        data, fieldnames = create_data_and_fieldnames(balance_list, data, currency_list)

        filename = './cryptocurrency_' + self.__exchange_name + '_' + dt_month + '.csv'

        write_csvfile(filename, fieldnames, data)

    def get_currency_summary(self):
        balance_list, total_assets, currency_list = self.get_currency_data()

        
