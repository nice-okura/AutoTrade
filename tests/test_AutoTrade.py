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
    def get_test_songiri_30days(self):
        print("### get_test_songiri_30days_fixture ###")
        df = self.load_csv2pd('tests/test_songiri_3days.csv')
        position_df = self.load_csv2pd('tests/test_songiri_3days_position.csv')

        yield({'df': df, 'position_df': position_df})

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
        logic = 0
        df = get_data_20210401_0404_fixture['df']
        df = self.at.calc_features(df)
        df['ma_diff'], df['GCDC_times'] = self.at.get_madata(df)
        # df['RSI'] = self.at.get_rsi(df)
        sim_df = self.at.simulate(df, logic, init_yen=100000, init_coin=100, price_decision_logic=0)

        assert sim_df['Coin'][-1] == 100
        assert sim_df['Profit'][-1] == 3653.634694403416
        assert sim_df['SimulateAsset'][-1] == 217194.0346944034


    def test_simulate_20210401_0404_logic1(self, get_data_20210401_0404_fixture):
        logic = 1
        df = get_data_20210401_0404_fixture['df']
        df = self.at.calc_features(df)
        df['ma_diff'], df['GCDC_times'] = self.at.get_madata(df)
        # df['RSI'] = self.at.get_rsi(df)
        sim_df = self.at.simulate(df, logic, init_yen=100000, init_coin=100, price_decision_logic=0)

        assert sim_df['Coin'][-1] == 100
        assert sim_df['Profit'][-1] == 3653.634694403416
        assert sim_df['SimulateAsset'][-1] == 217194.0346944034

    def test_get_BUYSELLprice_mode0(self):
        yen_price = 100
        coin_price = 20
        coin = 100
        yen = 10000
        mode = 0

        assert self.at.get_BUYSELLprice(yen_price, coin_price, coin, yen, mode) == 100


    def test_get_BUYSELLprice_mode1(self, get_madata_gcdc81_fixture):
        df = get_madata_gcdc81_fixture['df']
        oneline_df = df.iloc[-1:]
        yen_price = 100
        coin_price = 20
        coin = 100
        yen = 10000
        mode = 1

        assert self.at.get_BUYSELLprice(yen_price, coin_price, coin, yen, mode, oneline_df) == 405

    @pytest.mark.this
    def test_songiri(self, get_test_songiri_30days):
        df = get_test_songiri_30days['df']
        position_df = get_test_songiri_30days['position_df']
        tmp_df = df.iloc[-1:]
        coin_price = tmp_df['Close'][0]
        init_coin = 100
        coin = init_coin
        init_yen = 10000
        yen = init_yen
        price_decision_logic = 0

        df, position_df, coin, yen = self.at.songiri(df, position_df, coin_price, coin, yen, price_decision_logic, tmp_df)
        assert len(position_df) == 8
        assert yen == init_yen - self.at.param.BUY_PRICE
        assert coin == init_coin + self.at.param.BUY_PRICE/coin_price

    def test_songiri_simulate(self, get_test_songiri_30days):
        df = get_test_songiri_30days['df']
        sim_df = self.at.simulate(df, logic=1, init_yen=100000, init_coin=100, price_decision_logic=0)
