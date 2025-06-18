import pytest
import pandas as pd
from src.data_processing import process_data

def test_process_data():
    data = pd.DataFrame({'DIA': [100, 101, 102], 'SPY': [200, 201, 202]}, index=['2020-01-01', '2020-01-02', '2020-01-03'])
    # Convert index to DatetimeIndex
    data.index = pd.to_datetime(data.index)
    processed = process_data(data)
    assert not processed.isnull().any().any(), "Processed data contains NaN"