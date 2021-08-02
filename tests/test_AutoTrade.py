from AutoTrade import *
import pytest
from datetime import datetime
import pandas as pd
from CryptService import CryptService

def load_csv2pd(filename):
    df = pd.read_csv(filename)
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

"""

  *******************
  ***** T E S T *****
  *******************

"""
def test_get_ohlcv_5min():
    date = datetime.now()
    size = 1000
    df = get_ohlcv(date, size, '5min')

    assert df.size >= size

def test_get_ohlcv_1hour():
    date = datetime.now()
    size = 100
    df = get_ohlcv(date, size, '1hour')

    assert df.size >= size

# 4hourには対応していないので、例外が発生します
def test_get_ohlcv_4hour():
    date = datetime.now()
    size = 100

    with pytest.raises(Exception):
        df = get_ohlcv(date, size, '4hour')

def test_get_madata_times1(get_madata_times1_fixture):
    """
    GCDC_times が1となる場合
    """
    df = get_madata_times1_fixture['df']
    ma_diff, times_list = get_madata(df)

    assert times_list[-1] == 1

def test_get_madata_times81(get_madata_times81_fixture):
    """
    GCDC_times が81となる場合
    """
    df = get_madata_times81_fixture['df']
    ma_diff, times_list = get_madata(df)

    assert times_list[-1] == 81

def test_get_madata_madiff(get_madata_times81_fixture):
    """
    ma_diff が68222.3116となる場合
      2021-06-14 08:00:00+09:00,1009.391,1016.5,1001.021,1013.0,68222.3116,-7.083140000000185,81,77.22267757450163
    """
    df = get_madata_times81_fixture['df']

    ma_diff, times_list = get_madata(df)

    assert ma_diff.iloc()[-1] == -7.083140000000185

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
    mocker.patch.object(CryptService, 'post_order', return_value={'success': 1, 'data': {'start_amount': 100.0}})
    assert order(BUY, 1.0, 100.0) == 100.0

def test_order_sell(mocker):
    mocker.patch.object(CryptService, 'post_order', return_value={'success': 1, 'data': {'start_amount': 100.0}})
    assert order(SELL, 1.0, 100.0) == 100.0

def test_order_buy_error(mocker):
    mocker.patch.object(CryptService, 'post_order', return_value={'success': 0, 'data': 'モック'})
    assert order(BUY, 1.0, 100.0) == -1

def test_order_sell_error(mocker):
    mocker.patch.object(CryptService, 'post_order', return_value={'success': 0, 'data': 'モック'})
    assert order(SELL, 1.0, 100.0) == -1
