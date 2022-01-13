from AutoTrade import *
import pytest
from datetime import datetime, timedelta
import pandas as pd
from CryptService import CryptService

def load_csv2pd(filename):
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
def session_fixture():
    print("テスト全体の前処理")
    MA_short = 5  # 移動平均（短期）
    MA_long = 50  # 移動平均（長期）
    CANDLE_TYPE = '1hour'  # データ取得間隔
    PAIR = 'qtum_jpy'  # 対象通貨
    MA_times = 12  # コインを購入/売却金額する連続GC/DC回数
    BUY_PRICE = 400.0  # 購入金額(円)
    SELL_PRICE = 400.0  # 売却金額(円)
    RSI_SELL = 85.0  # 売りRSIボーダー
    RSI_BUY = 100.0 - RSI_SELL  # 買いRSIボーダー
    VOL_ORDER = 20000  # 取引する基準となる取引量(Volume)
    BUY = 1
    SELL = -1
    WEIGHT_OF_PRICE = 0.2  # 連続MA回数から購入金額を決めるときの重み

    yield
    print("テスト全体の後処理")


@pytest.fixture(scope='module', autouse=True)
def module_fixture():
    print("モジュールの前処理")
    yield
    print("モジュールの後処理")


@pytest.fixture(scope='class', autouse=True)
def class_fixture():
    print("クラスの前処理")
    yield
    print("クラスの後処理")


@pytest.fixture(scope='function', autouse=True)
def function_fixture():
    print("関数の前処理")
    yield
    print("関数の後処理")


@pytest.fixture(scope='function', autouse=True)
def get_madata_times1_fixture():
    test_df = load_csv2pd('tests/test_df_times1.csv')
    yield({'df': test_df})


@pytest.fixture(scope='function', autouse=True)
def get_madata_times81_fixture():
    print("get_madata_times81_fixture()")
    test_df = load_csv2pd('tests/test_df_times81.csv')
    yield({'df': test_df})


@pytest.fixture(scope='function', autouse=True)
def get_madata_gcdc1_fixture():
    test_df = load_csv2pd('tests/test_df_gcdc1.csv')
    yield({'df': test_df})


@pytest.fixture(scope='function', autouse=True)
def get_madata_gcdc81_fixture():
    print("### get_madata_gcdc81_fixture ###")
    test_df = load_csv2pd('tests/test_df_gcdc81.csv')

    yield({'df': test_df})


@pytest.fixture(scope='function', autouse=True)
def get_madata_rsiover80_fixture():
    print("### get_madata_rsiover80_fixture ###")
    test_df = load_csv2pd('tests/test_df_rsiover80.csv')

    yield({'df': test_df})


@pytest.fixture(scope='function', autouse=True)
def get_madata_rsiunder20_fixture():
    print("### get_madata_rsiunder20_fixture ###")
    test_df = load_csv2pd('tests/test_df_rsiunder20.csv')

    yield({'df': test_df})

@pytest.fixture(scope='function', autouse=True)
def get_data_20210401_0404_fixture():
    print("### get_data_20210401_0404_fixture ###")
    test_df = load_csv2pd('tests/data_20210401-0404.csv')

    yield({'df': test_df})

@pytest.fixture(scope='function', autouse=True)
def get_test_df_result_fixture():
    print("### get_test_df_result_fixture ###")
    test_df = load_csv2pd('tests/test_df_result.csv')

    yield({'df': test_df})

"""

  *******************
  ***** T E S T *****
  *******************

"""


def test_get_ohlcv_5min():
    date = datetime.now() - timedelta(days=1)
    size = 1000
    df = get_ohlcv(date, size, '5min')

    assert df.size >= size


def test_get_ohlcv_1hour():
    date = datetime.now() - timedelta(days=1)
    size = 100
    df = get_ohlcv(date, size, '1hour')

    assert df.size >= size


# 4hourには対応していないので、例外が発生します
def test_get_ohlcv_4hour():
    date = datetime.now() - timedelta(days=1)
    size = 100

    with pytest.raises(Exception):
        df = get_ohlcv(date, size, '4hour')


def test_get_madata_times1(get_madata_times1_fixture):
    """
    GCDC_times が1となる場合
    """
    df = get_madata_times1_fixture['df']
    ma_diff, times_list = get_madata(df)

    assert times_list[-1] == 11


def test_get_madata_times81(get_madata_times81_fixture):
    """
    GCDC_times が81となる場合
    """
    df = get_madata_times81_fixture['df']
    ma_diff, times_list = get_madata(df)

    assert times_list[-1] == 5

def test_get_madata_madiff(get_madata_times81_fixture):
    """
    ma_diff が68222.3116となる場合
      2021-06-14 08:00:00+09:00,1009.391,1016.5,1001.021,1013.0,68222.3116,-7.083140000000185,81,77.22267757450163
    """
    df = get_madata_times81_fixture['df']

    ma_diff, times_list = get_madata(df)

    assert ma_diff.iloc()[-1] == 50.50217999999984

def test_get_rsi_01(get_madata_times81_fixture):
    """
    rsiが77.22267757450163となる
    """
    df = get_madata_times81_fixture['df']
    rsi = get_rsi(df)

    assert rsi.iloc()[-1] == 77.22267757450163

def test_get_rsi_02(get_madata_times1_fixture):
    """
    rsiが69.09899987493397となる
    """
    df = get_madata_times1_fixture['df']
    rsi = get_rsi(df)

    assert rsi.iloc()[-1] == 69.09899987493397

def test_is_gcdc(get_madata_gcdc81_fixture):
    """
    連続GDCD回数が9の倍数
    """
    df = get_madata_gcdc81_fixture['df']

    assert is_gcdc(df, 9)

def test_isnot_gcdc(get_madata_gcdc1_fixture):
    """
    連続GDCD回数が9の倍数でない
    """
    df = get_madata_gcdc1_fixture['df']

    assert not is_gcdc(df, 9)

def test_buysell_by_rsi_sell(get_madata_rsiover80_fixture):
    """
    RSIから売りと判断される
    """
    df = get_madata_rsiover80_fixture['df']

    assert buysell_by_rsi(df) == -1

def test_buysell_by_rsi_buy(get_madata_rsiunder20_fixture):
    """
    RSIから買いと判断される
    """
    df = get_madata_rsiunder20_fixture['df']

    assert buysell_by_rsi(df) == 1

def test_buysell_by_rsi_stay(get_madata_gcdc1_fixture):
    """
    RSIから売りと判断される
    """
    df = get_madata_gcdc1_fixture['df']

    assert buysell_by_rsi(df) == 0

def test_buysell_by_vol(get_madata_gcdc1_fixture):
    """
    VOLが低いの取引しない
    """
    df = get_madata_gcdc1_fixture['df']

    assert not buysell_by_vol(df)

def test_buysell_by_vol(get_madata_gcdc81_fixture):
    """
    VOLが十分なので取引する
    """
    df = get_madata_gcdc81_fixture['df']

    assert buysell_by_vol(df)

def test_order_buy(mocker):
    mocker.patch('AutoTrade.DEBUG', False)
    mocker.patch.object(CryptService, 'post_order', return_value={'success': 1, 'data': {'start_amount': 100.0}})
    assert order(BUY, 1.0, 100.0) == 100.0

def test_order_sell(mocker):
    mocker.patch('AutoTrade.DEBUG', False)
    mocker.patch.object(CryptService, 'post_order', return_value={'success': 1, 'data': {'start_amount': 100.0}})
    assert order(SELL, 1.0, 100.0) == 100.0


def test_order_buy_error(mocker):
    mocker.patch.object(CryptService, 'post_order', return_value={'success': 0, 'data': 'モック'})
    assert order(BUY, 1.0, 100.0) == -1


def test_order_sell_error(mocker):
    mocker.patch.object(CryptService, 'post_order', return_value={'success': 0, 'data': 'モック'})
    assert order(SELL, 1.0, 100.0) == -1


def test_simulate_20210401_0404_logic0(get_data_20210401_0404_fixture):
    logic = 0
    df = get_data_20210401_0404_fixture['df']
    df['ma_diff'], df['GCDC_times'] = get_madata(df)
    df['rsi'] = get_rsi(df)
    sim_df = simulate(df, logic)

    assert sim_df['Coin'][-1] == 100.67201231804299
    assert sim_df['Profit'][-1] == 3663.809318778862
    assert sim_df['SimulateAsset'][-1] == 217204.20931877886


def test_simulate_20210401_0404_logic1(get_data_20210401_0404_fixture):
    logic = 1
    df = get_data_20210401_0404_fixture['df']
    df['ma_diff'], df['GCDC_times'] = get_madata(df)
    df['rsi'] = get_rsi(df)
    sim_df = simulate(df, logic, init_yen=100000, init_coin=100, price_decision_logic=0)

    assert sim_df['Coin'][-1] == 40.5112654518479
    assert sim_df['Profit'][-1] == 1469.7274683652795
    assert sim_df['SimulateAsset'][-1] == 146885.88746836528

def test_get_BUYSELLprice_mode0():
    yen_price = 100
    coin_price = 20
    mode = 0
    assert get_BUYSELLprice(yen_price, coin_price, mode) == 100


def test_get_BUYSELLprice_mode1(get_madata_gcdc81_fixture):
    df = get_madata_gcdc81_fixture['df']
    oneline_df = df.iloc[-1:]
    yen_price = 100
    cryptcoin_price = 20
    mode = 1
    assert get_BUYSELLprice(yen_price, cryptcoin_price, mode, oneline_df) == 405

def test_save_graph(get_test_df_result_fixture):
    df = get_test_df_result_fixture['df']
    save_gragh(df, "test_df_result.png")
