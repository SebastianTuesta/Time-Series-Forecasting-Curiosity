#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
utils.financial_data_preprocesing : a package to financial data acquisition and prepocesting .
"""
import numpy as np
import pandas as pd
from datetime import date, datetime as datetime_datetime
from pandas_datareader import DataReader as pdr_DataReader
import pywt

from typing import Optional, Tuple, List

class Financial_Data_Preprocesing:

    def __init__(self, ticker: str, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> None:
        """
        Parameters
        ----------
        ticker : string
            The short name of stock in Yahoo Finance!
        start : tuple
            The begin date of the historical values: (Year, Month, Day)
        end : tuple
            The end date of the historical values: (Year, Month, Day)
        """
        super().__init__()
        self.df: pdr_DataReader = None
        self.df_min_max: pd.DataFrame = None
        self.get_stock("^NYA", start = (2012,1,3), end = (2018,12,31))
        self.n = len(self.df)

    def fit(self):
        """
        Fit the default values to preprocessing
        """
        prev = []
        for col in self.df.columns:
            prev.append([
                col,
                min(self.df[col]),
                max(self.df[col])
            ])
        self.df_min_max = pd.DataFrame(prev, columns = ["Target", "Min", "Max"])

    def transform(self):
        """
        Preprecessing per-se
        """
        self.min_max_transformation()

        for col in self.df.columns:
            self.wavelet_denoising(col = col)
        
        self.add_sma()
        self.add_ema()
        self.add_obv()
        self.add_macd()
        self.add_rsi()

    def get_stock(self, ticker: str, start: Tuple[int, int, int], end: Tuple[int, int, int]):
        r"""Return a dataframe of historical value for a particular stock.

        Parameters
        ----------
        ticker : string
            The short name of stock in Yahoo Finance!
        start : tuple
            The begin date of the historical values: (Year, Month, Day)
        end : tuple
            The end date of the historical values: (Year, Month, Day)
        """
        start_date = datetime_datetime(*start)
        end_date = datetime_datetime(*end)

        self.df = pdr_DataReader(ticker, 'yahoo',start_date, end_date)

    def add_sma(self, days = 30):
        r"""Return a dataframe of historical value for a particular stock adding the SMA values.

        Parameters
        ----------
        days : int
            Parameter used in the sma formula. Default = 30

        """
        for col in self.df.columns:
            #SMA
            self.df['SMA_{0}_{1}'.format(col, days)] = self.df[col].rolling(window = days).mean()

    def add_ema(self, days = 30):
        r"""Return a dataframe of historical value for a particular stock adding the EMA values.

        Parameters
        ----------
        days : int
            Parameter used in the ema formula. Default = 30
        """
        for col in self.df.columns:
            self.df['EMA_{0}_{1}'.format(col, days)] = pd.Series.ewm(self.df[col], span = days).mean()

    def add_obv(self):
        r"""Return a dataframe of historical value for a particular stock adding the OBV values.
        """
        self.df['OBV'] =  (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()

    def add_macd(self):
        r"""Return a dataframe of historical value for a particular stock adding the MACD values.
        """
        exp1 = self.df["Adj Close"].ewm(span = 26, adjust=False).mean()
        exp2 = self.df["Adj Close"].ewm(span = 13, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2

    def add_rsi(self):
        r"""Return a dataframe of historical value for a particular stock adding the RSI values.
        """
        
        delta = self.df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up/ema_down

        self.df['RSI'] = 100 - (100/(1+rs))

    def wavelet_denoising(self, col: str, wavelet:Optional[str]='db4', level:Optional[int]=1):
        r"""Return a time series denoise using wavelet transform

        Parameters
        ----------
        col : string
            Name of the column
        wavelet: string
            Wavelet Function name
        level: int
            Wavelet level
        """
        coeff = pywt.wavedec(self.df[col], wavelet, mode="per")
        madev = np.mean(np.absolute(coeff[-level] - np.mean(coeff[-level], axis = None)), axis = None)
        sigma = (1/0.6745) * madev
        uthresh = sigma * np.sqrt(2 * np.log(len(self.df[col])))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        return pywt.waverec(coeff, wavelet, mode='per')

    def min_max_transformation(self):
        r"""Return a Data frame normalized
        """
        for col in self.df.columns:
            _min_ = self.df_min_max[self.df_min_max["Target"] == col]["Min"].values[0]
            _max_ = self.df_min_max[self.df_min_max["Target"] == col]["Max"].values[0]
            
            self.df[col] = ( self.df[col].sub(_min_, axis=0) )/( _max_ - _min_ )

    def min_max_untransformation(self, x, col):
        """
        From min-max transformation to real value
        """
        _min_ = self.df_min_max[self.df_min_max["Target"] == col]["Min"].values[0]
        _max_ = self.df_min_max[self.df_min_max["Target"] == col]["Max"].values[0]
                    
        return ( _max_ - _min_ )*np.asarray(x) + _min_