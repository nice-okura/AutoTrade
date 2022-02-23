from AutoTrade import AutoTrade, Parameter, BUY, SELL
import pytest
from datetime import datetime, timedelta
import pandas as pd
from CryptService import CryptService

class TestAutoTrade:
    def load_csv2pd(self, filename):
        df = pd.read_csv(filename, parse_dates=[0])
        df = df.set_index('Date')
        df = df.astype(float)

        return df


    """
      *************************
      ***** F I X T U R E *****
      *************************
    """


    @pytest.fixture(scope='session', autouse=True)
    def session_fixture(self):
        print("テスト全体の前処理")
        yield
        print("テスト全体の後処理")


    @pytest.fixture(scope='module', autouse=True)
    def module_fixture(self):
        print("モジュールの前処理")
        yield
        print("モジュールの後処理")


    @pytest.fixture(scope='class', autouse=True)
    def class_fixture(self):
        print("クラスの前処理")
        yield
        print("クラスの後処理")


    @pytest.fixture(scope='function', autouse=True)
    def function_fixture(self):
        param = Parameter()
        self.at = AutoTrade(param)
        yield

    @pytest.fixture(scope='class', autouse=True)
    def get_madata_times1_fixture(self):
        test_df = self.load_csv2pd('tests/test_df_times1.csv')
        yield({'df': test_df})


    # @pytest.fixture(scope='function', autouse=True)
    @pytest.fixture(scope='class')
    def get_madata_times81_fixture(self):
        print("get_madata_times81_fixture()")
        test_df = self.load_csv2pd('tests/test_df_times81.csv')
        yield({'df': test_df})


    @pytest.fixture(scope='class', autouse=True)
    def get_madata_gcdc1_fixture(self):
        test_df = self.load_csv2pd('tests/test_df_gcdc1.csv')
        yield({'df': test_df})


    @pytest.fixture(scope='class', autouse=True)
    def get_madata_gcdc81_fixture(self):
        print("### get_madata_gcdc81_fixture ###")
        test_df = self.load_csv2pd('tests/test_df_gcdc81.csv')

        yield({'df': test_df})


    @pytest.fixture(scope='class', autouse=True)
    def get_madata_rsiover80_fixture(self):
        print("### get_madata_rsiover80_fixture ###")
        test_df = self.load_csv2pd('tests/test_df_rsiover80.csv')

        yield({'df': test_df})


    @pytest.fixture(scope='class', autouse=True)
    def get_madata_rsiunder20_fixture(self):
        print("### get_madata_rsiunder20_fixture ###")
        test_df = self.load_csv2pd('tests/test_df_rsiunder20.csv')

        yield({'df': test_df})

    @pytest.fixture(scope='class', autouse=True)
    def get_data_20210401_0404_fixture(self):
        print("### get_data_20210401_0404_fixture ###")
        test_df = self.load_csv2pd('tests/data_20210401-0404.csv')

        yield({'df': test_df})

    @pytest.fixture(scope='class', autouse=True)
    def get_test_df_result_fixture(self):
        print("### get_test_df_result_fixture ###")
        test_df = self.load_csv2pd('tests/test_df_result.csv')

        yield({'df': test_df})

    @pytest.fixture(scope='class', autouse=True)
    def get_test_songiri_7days_ohlcv(self):
        print("### get_test_songiri_7days_ohlcv ###")
        df = self.load_csv2pd('tests/test_songiri_7days_ohlcv.csv')
        position_df = self.load_csv2pd('tests/test_songiri_7days_position.csv')

        yield({'df': df, 'position_df': position_df})

    @pytest.fixture(scope='class', autouse=True)
    def get_test_songiri_1days(self):
        print("### get_test_songiri_1days_fixture ###")
        df = self.load_csv2pd('tests/test_songiri_1days.csv')
        position_df = self.load_csv2pd('tests/test_songiri_1days_position.csv')

        yield({'df': df, 'position_df': position_df})

    @pytest.fixture(scope='class', autouse=True)
    def get_test_songiri_3days_ohlcv(self):
        print("### get_test_songiri_3days_ohlcv ###")
        df = self.load_csv2pd('tests/test_songiri_3days_ohlcv.csv')

        yield({'df': df})
    """

      *******************
      ***** T E S T *****
      *******************

    """


    def test_get_ohlcv_5min(self):
        date = datetime.now() - timedelta(days=1)
        size = 1000
        df = self.at.get_ohlcv(date, size, '5min')

        assert df.size >= size


    def test_get_ohlcv_1hour(self):
        date = datetime.now() - timedelta(days=1)
        size = 100
        df = self.at.get_ohlcv(date, size, '1hour')

        assert df.size >= size


    # 4hourには対応していないので、例外が発生します
    def test_get_ohlcv_4hour(self):
        date = datetime.now() - timedelta(days=1)
        size = 100

        with pytest.raises(Exception):
            df = self.at.get_ohlcv(date, size, '4hour')


    def test_get_madata_times1(self, get_madata_times1_fixture):
        """
        GCDC_times が1となる場合
        """
        df = get_madata_times1_fixture['df']
        ma_diff, times_list = self.at.get_madata(df)

        assert times_list[-1] == 11


    def test_get_madata_times81(self, get_madata_times81_fixture):
        """
        GCDC_times が81となる場合
        """
        df = get_madata_times81_fixture['df']
        ma_diff, times_list = self.at.get_madata(df)

        assert times_list[-1] == 5


    def test_get_madata_madiff(self, get_madata_times81_fixture):
        """
        ma_diff が68222.3116となる場合
          2021-06-14 08:00:00+09:00,1009.391,1016.5,1001.021,1013.0,68222.3116,-7.083140000000185,81,77.22267757450163
        """
        df = get_madata_times81_fixture['df']

        ma_diff, times_list = self.at.get_madata(df)

        assert ma_diff.iloc()[-1] == 50.50217999999984

    # def test_get_rsi_01(self, get_madata_times81_fixture):
    #     """
    #     rsiが77.22267757450163となる
    #     """
    #     df = get_madata_times81_fixture['df']
    #     rsi = self.at.get_rsi(df)
    #
    #     assert rsi.iloc()[-1] == 77.22267757450163
    #
    # def test_get_rsi_02(self, get_madata_times1_fixture):
    #     """
    #     rsiが69.09899987493397となる
    #     """
    #     df = get_madata_times1_fixture['df']
    #     rsi = self.at.get_rsi(df)
    #
    #     assert rsi.iloc()[-1] == 69.09899987493397
    def test_is_gcdc(self, get_madata_gcdc81_fixture):
        """
        連続GDCD回数が9の倍数
        """
        df = get_madata_gcdc81_fixture['df']

        assert self.at.is_gcdc(df, 9)


    def test_isnot_gcdc(self, get_madata_gcdc1_fixture):
        """
        連続GDCD回数が9の倍数でない
        """
        df = get_madata_gcdc1_fixture['df']

        assert not self.at.is_gcdc(df, 9)


    def test_buysell_by_rsi_sell(self, get_madata_rsiover80_fixture):
        """
        RSIから売りと判断される
        """
        df = get_madata_rsiover80_fixture['df']

        assert self.at.buysell_by_rsi(df) == -1


    def test_buysell_by_rsi_buy(self, get_madata_rsiunder20_fixture):
        """
        RSIから買いと判断される
        """
        df = get_madata_rsiunder20_fixture['df']

        assert self.at.buysell_by_rsi(df) == 1


    def test_buysell_by_rsi_stay(self, get_madata_gcdc1_fixture):
        """
        RSIから売りと判断される
        """
        df = get_madata_gcdc1_fixture['df']

        assert self.at.buysell_by_rsi(df) == 0


    def test_buysell_by_vol_1(self, get_madata_gcdc1_fixture):
        """
        VOLが低いの取引しない
        """
        param = Parameter(vol_order=50000)
        self.at = AutoTrade(param)

        df = get_madata_gcdc1_fixture['df']

        assert not self.at.buysell_by_vol(df)


    def test_buysell_by_vol_2(self, get_madata_gcdc81_fixture):
        """
        VOLが十分なので取引する
        """
        param = Parameter(vol_order=50000)
        self.at = AutoTrade(param)

        df = get_madata_gcdc81_fixture['df']

        assert self.at.buysell_by_vol(df)


    def test_order_buy(self, mocker):
        mocker.patch('AutoTrade.DEBUG', False)
        mocker.patch.object(CryptService, 'post_order', return_value={'success': 1, 'data': {'start_amount': 100.0}})
        assert self.at.order(BUY, 1.0, 100.0) == 100.0


    def test_order_sell(self, mocker):
        mocker.patch('AutoTrade.DEBUG', False)
        mocker.patch.object(CryptService, 'post_order', return_value={'success': 1, 'data': {'start_amount': 100.0}})
        assert self.at.order(SELL, 1.0, 100.0) == 100.0


    def test_order_buy_error(self, mocker):
        mocker.patch.object(CryptService, 'post_order', return_value={'success': 0, 'data': 'モック'})
        assert self.at.order(BUY, 1.0, 100.0) == -1


    def test_order_sell_error(self, mocker):
        mocker.patch.object(CryptService, 'post_order', return_value={'success': 0, 'data': 'モック'})
        assert self.at.order(SELL, 1.0, 100.0) == -1


    def test_simulate_20210401_0404_logic0(self, get_data_20210401_0404_fixture):
        """
        logic = 0のテスト
        損切は実施しない
        """
        param = Parameter(logic=0, songiri=False)
        self.at = AutoTrade(param)

        df = get_data_20210401_0404_fixture['df']
        df = self.at.calc_features(df)
        df['ma_diff'], df['GCDC_times'] = self.at.get_madata(df)
        # df['RSI'] = self.at.get_rsi(df)
        sim_df = self.at.simulate(df, init_yen=100000, init_coin=100)

        assert sim_df['Coin'][-1] == 100
        assert sim_df['Profit'][-1] == 3653.634694403416
        assert sim_df['SimulateAsset'][-1] == 217194.0346944034


    def test_simulate_20210401_0404_logic1(self, get_data_20210401_0404_fixture):
        """
        logic = 1のテスト
        損切は実施しない
        """
        param = Parameter(logic=1, songiri=False)
        self.at = AutoTrade(param)

        df = get_data_20210401_0404_fixture['df']
        df = self.at.calc_features(df)
        df['ma_diff'], df['GCDC_times'] = self.at.get_madata(df)
        # df['RSI'] = self.at.get_rsi(df)
        sim_df = self.at.simulate(df, init_yen=100000, init_coin=100)

        assert sim_df['Coin'][-1] == 100
        assert sim_df['Profit'][-1] == 3653.634694403416
        assert sim_df['SimulateAsset'][-1] == 217194.0346944034


    def test_get_BUYSELLprice_mode0(self):
        """
        価格決定ロジック = 0
        基本購入価格 100円
        に設定してテスト
        """
        param = Parameter(price_decision_logic=0, buy_price=100)
        self.at = AutoTrade(param=param)

        yen_price = 100
        coin_price = 20
        coin = 100
        yen = 10000
        bs_price = self.at.get_BUYSELLprice(self.at.param.BUY_PRICE, coin_price, coin, yen)

        assert bs_price == 100


    def test_get_BUYSELLprice_mode1(self, get_madata_gcdc81_fixture):
        """
        価格決定ロジック = 1
        基本購入価格 100円
        に設定してテスト
        """
        param = Parameter(price_decision_logic=1, buy_price=100)
        self.at = AutoTrade(param=param)

        # self.at.param.price_decision_logic = 1
        df = get_madata_gcdc81_fixture['df']
        oneline_df = df.iloc[-1:]
        coin_price = 20
        coin = 100
        yen = 10000
        bs_price = self.at.get_BUYSELLprice(self.at.param.BUY_PRICE, coin_price, coin, yen, oneline_df)

        assert bs_price == 405


    def test_songiri(self, get_test_songiri_7days_ohlcv):
        df = get_test_songiri_7days_ohlcv['df']
        position_df = get_test_songiri_7days_ohlcv['position_df']
        tmp_df = df.iloc[-1:]
        coin_price = tmp_df['Close'][0]
        init_coin = 100
        coin = init_coin
        init_yen = 100000
        yen = init_yen
        price_decision_logic = 0

        df, position_df, coin, yen = self.at.songiri(df, position_df, coin_price, coin, yen, tmp_df)
        assert len(position_df) == 8
        assert yen == init_yen - self.at.param.BUY_PRICE
        assert coin == init_coin + self.at.param.BUY_PRICE/coin_price


    def test_songiri_1days(self, get_test_songiri_1days):
        """
        損切する場合のテスト
        """
        # 売買価格を100円にしてテスト
        param = Parameter(buy_price=100.0, sell_price=100.0, price_decision_logic=0)
        self.at = AutoTrade(param)

        df = get_test_songiri_1days['df']
        position_df = get_test_songiri_1days['position_df']
        tmp_df = df.iloc[-1:]
        coin_price = tmp_df['Close'][0]
        init_coin = 99.94117647
        coin = init_coin
        init_yen = 100100.0
        yen = init_yen
        price_decision_logic = 0

        df, position_df, coin, yen = self.at.songiri(df, position_df, coin_price, coin, yen, tmp_df)
        assert len(position_df) == 0
        assert yen == 100000.0
        assert round(coin, 5) == round(99.99436796,5)


    def test_songiri_1days_nosongiri(self, get_test_songiri_1days):
        """
        損切しない場合のテスト
        """
        # 売買価格を100円にしてテスト
        param = Parameter(buy_price=100.0, sell_price=100.0, price_decision_logic=0)
        self.at = AutoTrade(param)

        df = get_test_songiri_1days['df']
        # 損切すべき時刻のデータを削除することで
        # 損切しないテストデータを作る
        df = df.drop(df.index[[-1]])

        position_df = get_test_songiri_1days['position_df']
        tmp_df = df.iloc[-1:]
        coin_price = tmp_df['Close'][0]
        init_coin = 99.94117647
        coin = init_coin
        init_yen = 100100.0
        yen = init_yen
        price_decision_logic = 0

        df, position_df, coin, yen = self.at.songiri(df, position_df, coin_price, coin, yen, tmp_df)
        assert len(position_df) == 1
        assert yen == init_yen
        assert round(coin, 5) == round(init_coin, 5)


    def test_songiri_simulate_3day(self, get_test_songiri_3days_ohlcv):
        df = get_test_songiri_3days_ohlcv['df']
        # 損切しやすいように、SONGIRI_PERCを0.01(1%)にしてテスト
        param = Parameter(songiri_perc=0.01)
        self.at = AutoTrade(param)

        df = self.at.set_ma(df)
        df = self.at.calc_features(df)

        sim_df = self.at.simulate(df, init_yen=100000, init_coin=100)

        buy_df = sim_df[sim_df['BUYSELL']==1]
        sell_df = sim_df[sim_df['BUYSELL']==-1]

        assert len(buy_df) == 1
        assert len(sell_df) == 1
        assert buy_df.index[0].strftime("%Y%m%d %H:%M:%S") == '20211204 14:00:00'
        assert sell_df.index[0].strftime("%Y%m%d %H:%M:%S") == '20211204 15:00:00'


    def test_songiri_simulate_3day_logic4(self, get_test_songiri_7days_ohlcv):
        df = get_test_songiri_7days_ohlcv['df']
        # logic 4 にしてテスト
        param = Parameter(logic=4)
        self.at = AutoTrade(param)

        df = self.at.set_ma(df)
        df = self.at.calc_features(df)

        sim_df = self.at.simulate(df, init_yen=100000, init_coin=100)

        songiribuy_df = sim_df[(sim_df['BUYSELL']==1) & (sim_df['Songiri']==True)]
        songirisell_df = sim_df[(sim_df['BUYSELL']==-1) & (sim_df['Songiri']==True)]

        assert len(songiribuy_df) == 0
        assert songirisell_df.index[0].strftime("%Y-%m-%d %H:%M:%S") == '2021-12-04 09:00:00'
        assert songirisell_df.index[1].strftime("%Y-%m-%d %H:%M:%S") == '2021-12-04 12:00:00'
        assert songirisell_df.index[2].strftime("%Y-%m-%d %H:%M:%S") == '2021-12-04 20:00:00'
        assert songirisell_df.index[3].strftime("%Y-%m-%d %H:%M:%S") == '2021-12-06 15:00:00'

    def test_simulate_logic10(self, get_test_songiri_7days_ohlcv):
        logic = 10
        df = get_test_songiri_7days_ohlcv['df']
        df = self.at.set_ma(df)
        df = self.at.calc_features(df)
        print(df.head())
        df = df.dropna()
        print(df.head())

        # df['RSI'] = self.at.get_rsi(df)
        sim_df = self.at.simulate(df, init_yen=100000, init_coin=100)

    def test_ml(self, get_test_songiri_3days_ohlcv):
        df = get_test_songiri_3days_ohlcv['df']

        df = self.at.set_ma(df)
        df = self.at.calc_features(df)
        df = df.dropna()

        df = self.at.set_y(df)

    def test_simulate_load_model(self, get_test_songiri_3days_ohlcv):
        df = get_test_songiri_3days_ohlcv['df']
        param = Parameter(ml_model="./test_model.pkl")
        self.at = AutoTrade(param)

        df = self.at.set_ma(df)
        df = self.at.calc_features(df)
        df = df.dropna()

        sim_df = self.at.simulate(df, init_yen=100000, init_coin=100)

        print(sim_df['BUYSELL'].tail())
