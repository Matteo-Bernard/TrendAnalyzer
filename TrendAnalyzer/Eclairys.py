import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import tempfile

class Eclairys:
    def __init__(self, history, short_period, long_period):
        """
        Initialize the StockAnalyzer with historical stock data and periods.

        Parameters:
        - history: Series containing historical stock prices.
        - short_period: List of 3 integers representing short periods.
        - long_period: List of 3 integers representing long periods.
        """
        self.history = history #.resample('D').ffill()
        self.short_period = short_period
        self.long_period = long_period

    def moving_average(self):
        """
        Calculate moving averages based on historical stock data.

        Returns:
        - MA_df: DataFrame containing moving averages for each period.
        """
        MA_df = pd.DataFrame(index=self.history.index)
        MA_df[self.history.name] = self.history

        for period in self.short_period + self.long_period:
            if len(self.history) < period:
                MA_df[f'MA {period}'] = np.nan
            else:
                MA = self.history.rolling(period).mean()
                MA_df[f'MA {period}'] = MA

        return MA_df

    def gps(self):
        """
        Generate GPS indicators based on historical stock data.

        Returns:
        - GPS: Series containing GPS indicators.
        """
        MAD_df = pd.DataFrame()
        for period in self.short_period + self.long_period:
            if len(self.history) < period:
                MAD_df[f'MAD {period}'] = np.nan
            else:
                MA = self.history.rolling(period).mean()
                MAD = MA.diff(5)
                MAD_df[f'MAD {period}'] = MAD

        BMAD = (MAD_df >= 0).astype(int)
        short_columns = [f'MAD {p}' for p in self.short_period]
        long_columns = [f'MAD {p}' for p in self.long_period]

        SBMAD = BMAD[short_columns].sum(axis=1)
        LBMAD = BMAD[long_columns].sum(axis=1)

        conditions = [
            (SBMAD >= 2) & (LBMAD >= 2),
            (SBMAD <= 2) & (LBMAD <= 2),
            (SBMAD >= 2) & (LBMAD <= 2),
            (SBMAD <= 2) & (LBMAD >= 2)
        ]
        choices = ['A', 'B', 'C', 'P']
        GPS = pd.Series(np.select(conditions, choices, default="P"), index=self.history.index)

        return GPS

    def grade(self):
        """
        Calculate the GRADE score based on historical stock data.

        Returns:
        - GRADE: Series containing GRADE scores.
        """
        MAD_df = pd.DataFrame()
        for period in self.short_period + self.long_period:
            if len(self.history) < period:
                MAD_df[f'MAD {period}'] = np.nan
            else:
                MA = self.history.rolling(period).mean()
                MAD = MA.diff(5)
                MAD_df[f'MAD {period}'] = MAD

        MADN_df = pd.DataFrame(index=MAD_df.index)
        rolling_max = MAD_df.rolling(200, min_periods=1).max()
        rolling_min = MAD_df.rolling(200, min_periods=1).min()
        MADN_df = 100 * (MAD_df - rolling_min) / (rolling_max - rolling_min)
        MADN_df = MADN_df.replace([np.inf, -np.inf], 0).fillna(0)

        GRADE = MADN_df.mean(axis=1)

        return GRADE

    def plot(self, start=None, end=None, show=True):
        """
        Plot the stock analysis including moving averages, GRADE, and GPS.

        Parameters:
        - start: Start date for the plot.
        - end: End date for the plot.
        - show: Whether to display the plot or save it as an image.

        Returns:
        - eclairys_chart: Path to the saved chart image if show is False.
        """
        # Calculate necessary data
        MA_df = self.moving_average()
        GPS = self.gps()
        GRADE = self.grade()

        # Combine data into a single DataFrame
        df = pd.concat([MA_df, GRADE.rename('GRADE'), GPS.rename('Absolute GPS')], axis=1)

        # Filter data based on start and end dates
        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]

        # Plotting
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1, 0.25], 'hspace': 0.3})

        stock_ticker = df.columns[0]

        # 1st plot: Stock Price and Moving Averages
        ax1.plot(df.index, df[stock_ticker], label=stock_ticker, color='black', linewidth=1)
        for period in self.short_period + self.long_period:
            ax1.plot(df.index, df[f'MA {period}'], label=f'MA {period}', linestyle='--', linewidth=0.85)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax1.set_title(f'{self.history.name} Price and Moving Averages', fontsize=10)
        ax1.grid(True, alpha=0.5)

        # 2nd plot: GRADE
        ax2.plot(df.index, df['GRADE'], color='black')
        ax2.axhline(y=75, color='green', linewidth=0.75)
        ax2.axhline(y=25, color='red', linewidth=0.75)
        ax2.set_title('GRADE', fontsize=10)
        ax2_legend_elements = [Line2D([0], [0], color='black', lw=1, label='Grade'),
                               Line2D([0], [0], color='green', lw=0.75, label='Buy'),
                               Line2D([0], [0], color='red', lw=0.75, label='Sell')]
        ax2.legend(handles=ax2_legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax2.grid(True, alpha=0.5)

        # 3rd plot: Absolute GPS with color bands
        gps_colors = {'A': 'green', 'B': 'red', 'P': 'blue', 'C': 'orange'}
        for grade, color in gps_colors.items():
            mask = df['Absolute GPS'] == grade
            ax3.fill_between(df.index, 0, 1, where=mask, color=color, alpha=0.5, transform=ax3.get_xaxis_transform())
        ax3.set_title('Absolute GPS', fontsize=10)
        ax3.set_yticks([])

        # Add GPS legend
        ax3_legend_elements = [Patch(facecolor='green', label='A', alpha=0.5),
                               Patch(facecolor='red', label='B', alpha=0.5),
                               Patch(facecolor='blue', label='P', alpha=0.5),
                               Patch(facecolor='orange', label='C', alpha=0.5)]
        ax3.legend(handles=ax3_legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

        plt.tight_layout()

        if show:
            plt.show()
        else:
            # Save the plot to a file
            eclairys_chart = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(eclairys_chart.name, format='png', bbox_inches='tight')
            plt.close()

            return eclairys_chart.name
