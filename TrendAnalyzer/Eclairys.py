import yfinance as yf
import numpy as np
import pandas as pd

class eclairys:
    def __init__(self, stock_ticker, sector_ticker, market_ticker, short_period=[5, 10, 20], long_period=[50, 100, 200]):
        self.stock_ticker = stock_ticker
        self.sector_ticker = sector_ticker
        self.market_ticker = market_ticker
        self.short_period = short_period
        self.long_period = long_period
        self.period_list = short_period + long_period
        self.stock_history = None
        self.market_history = None
        self.sector_history = None
        self.market_relative_history = None
        self.sector_relative_history = None

    def fetch_data(self):
        """Fetch historical data for the stock and market."""
        self.stock_history = yf.Ticker(self.stock_ticker).history('max')['Close'].tz_localize(None).dropna()
        self.stock_history.name = self.stock_ticker
        self.market_history = yf.Ticker(self.market_ticker).history('max')['Close'].tz_localize(None).dropna()
        self.market_relative_history = self.stock_history / self.market_history
        self.sector_history = yf.Ticker(self.sector_ticker).history('max')['Close'].tz_localize(None).dropna()
        self.sector_relative_history = self.stock_history / self.sector_history

    def calculate_moving_averages(self, data):
        """Calculate moving averages and moving average differentials."""
        MA_df, MAD_df = pd.DataFrame(), pd.DataFrame()
        for period in self.period_list:
            if len(data) < period:
                MA_df[f'MA {period}'] = np.nan
                MAD_df[f'MAD {period}'] = np.nan
            else:
                MA = data.rolling(period).mean()
                MA_df[f'MA {period}'] = MA
                MAD = MA.diff(5)
                MAD_df[f'MAD {period}'] = MAD
        return MA_df, MAD_df

    def normalize_mad(self, MAD_df):
        """Normalize MAD using a rolling window."""
        MADN_df = pd.DataFrame(index=MAD_df.index)
        rolling_max = MAD_df.rolling(200, min_periods=1).max()
        rolling_min = MAD_df.rolling(200, min_periods=1).min()
        MADN_df = 100 * (MAD_df - rolling_min) / (rolling_max - rolling_min)
        MADN_df = MADN_df.replace([np.inf, -np.inf], 0).fillna(0)
        return MADN_df

    def calculate_grade(self, MADN_df):
        """Calculate the GRADE score as the mean of normalized MAD values."""
        GRADE = MADN_df.mean(axis=1)
        return GRADE

    def generate_gps(self, MAD_df, short_period, long_period):
        """Generate GPS indicators based on booleanized MAD values."""
        BMAD = (MAD_df >= 0).astype(int)
        
        # Créer les noms de colonnes pour les périodes courtes et longues
        short_columns = [f'MAD {p}' for p in short_period]
        long_columns = [f'MAD {p}' for p in long_period]
        
        # Calculer la somme des périodes courtes et longues
        SBMAD = BMAD[short_columns].sum(axis=1)
        LBMAD = BMAD[long_columns].sum(axis=1)

        # Générer les indicateurs GPS en fonction des conditions
        conditions = [
            (SBMAD >= 2) & (LBMAD >= 2),
            (SBMAD < 2) & (LBMAD < 2),
            (SBMAD >= 2) & (LBMAD < 2),
            (SBMAD < 2) & (LBMAD >= 2)
        ]
        choices = ['A', 'B', 'C', 'P']
        GPS = pd.Series(np.select(conditions, choices, default=None), index=MAD_df.index)
        return GPS

    def run_analysis(self):
        """Run the entire analysis and return the output DataFrame."""
        self.fetch_data()
        
        # Calculate indicators for the stock
        stock_MA_df, stock_MAD_df = self.calculate_moving_averages(self.stock_history)
        stock_MADN_df = self.normalize_mad(stock_MAD_df)
        stock_GRADE = self.calculate_grade(stock_MADN_df)
        stock_GPS = self.generate_gps(stock_MAD_df, self.short_period, self.long_period)

        # Combine stock indicators into a DataFrame
        stock_output = pd.concat([
            pd.DataFrame(self.stock_history),
            stock_MA_df,
            pd.DataFrame(stock_GRADE, columns=['GRADE']),
            pd.DataFrame(stock_GPS, columns=['Absolute GPS'])
        ], axis=1)

        # Calculate indicators for the relative trend on sector
        sector_MA_df, sector_MAD_df = self.calculate_moving_averages(self.sector_relative_history)
        sector_GPS = self.generate_gps(sector_MAD_df, self.short_period, self.long_period)
        stock_output['Sector GPS'] = sector_GPS

        # Calculate indicators for the relative trend on market
        market_MA_df, market_MAD_df = self.calculate_moving_averages(self.market_relative_history)
        market_GPS = self.generate_gps(market_MAD_df, self.short_period, self.long_period)
        stock_output['Market GPS'] = market_GPS

        return stock_output
    

