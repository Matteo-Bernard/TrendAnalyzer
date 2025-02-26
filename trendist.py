import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import tempfile

class Trends:
    def __init__(self, history, threshold, timeperiod='D'):
        # Resample the history data according to the specified time period
        self.history = history.resample(timeperiod).last()
        self.threshold = threshold
        self.ticker = history.name
        self.timeperiod = timeperiod
        self.trend = pd.DataFrame()

    def fit(self):
        history = self.history
        threshold = self.threshold

        # Check if history is empty or too short
        if len(history) < 2:
            raise ValueError("History data is too short to identify trends.")

        # Initialize DataFrame to store trend characteristics
        trend = pd.DataFrame(columns=['Ticker', 'First', 'Last', 'Return', 'Length', 'Way'])
        d0 = history.index[0]

        # Iterate through each date in the history to calculate trends
        for d in history.index[1:]:
            p = history.loc[d]

            # Calculate max and min prices within the current window
            pmax = history.loc[d0:d].max()
            dmax = history.loc[d0:d].idxmax()
            pmin = history.loc[d0:d].min()
            dmin = history.loc[d0:d].idxmin()

            # Calculate return based on the previous max or min
            if dmin < dmax:
                r = (p - pmax) / pmax
            else:
                r = (p - pmin) / pmin

            # Filter trends based on the threshold
            if abs(r) > threshold:
                new_trend = pd.DataFrame({
                    'Ticker': [self.ticker],
                    'First': [d0],
                    'Last': [dmax if dmin < dmax else dmin],
                    'Return': [r],
                    'Length': [(dmax - d0).days if dmin < dmax else (dmin - d0).days],
                    'Way': ['Up' if r > 0 else 'Down']
                })
                trend = pd.concat([trend, new_trend], ignore_index=True)
                d0 = dmax if dmin < dmax else dmin

        # Merge adjacent trends of the same direction
        trend = self._merge_adjacent_trends(trend)

        # Calculate the current trend
        last_trend = self._calculate_current_trend(trend, history)
        trend = pd.concat([trend, last_trend], ignore_index=True)

        self.trend = trend
        return self

    def _merge_adjacent_trends(self, trend):
        """
        Merge adjacent trends of the same direction.
        """
        if len(trend) == 0:
            return trend

        merged_trend = pd.DataFrame(columns=trend.columns)
        prev_row = trend.iloc[0]

        for i in range(1, len(trend)):
            current_row = trend.iloc[i]
            if prev_row['Way'] == current_row['Way']:
                # Merge with the previous trend
                prev_row['Last'] = current_row['Last']
                prev_row['Return'] = (
                    (self.history[current_row['Last']] - self.history[prev_row['First']]) /
                    self.history[prev_row['First']]
                )
                prev_row['Length'] = (current_row['Last'] - prev_row['First']).days
            else:
                # Add the previous trend to the merged DataFrame
                merged_trend = pd.concat([merged_trend, pd.DataFrame([prev_row])], ignore_index=True)
                prev_row = current_row

        # Add the last trend
        merged_trend = pd.concat([merged_trend, pd.DataFrame([prev_row])], ignore_index=True)

        return merged_trend

    def _calculate_current_trend(self, trend, history):
        """
        Calculate the current trend based on the last identified trend.
        """
        if len(trend) == 0:
            first = history.index[0]
        else:
            first = trend.iloc[-1]['Last']

        last = history.index[-1]
        current_return = (history[last] - history[first]) / history[first]

        return pd.DataFrame([{
            'Ticker': self.ticker,
            'First': first,
            'Last': last,
            'Return': current_return,
            'Length': (last - first).days,
            'Way': 'Up' if current_return > 0 else 'Down'
        }])

    def get_proba(self):
        history = self.history
        trend = self.trend

        # Calcul des caractéristiques de la séquence en cours 
        last_length = trend.iloc[-1]['Length']
        last_return = trend.iloc[-1]['Return']
        way = trend.iloc[-1]['Way']
        data = trend['Length'].to_list()

        # Ajustement des paramètres de la distribution de Pareto aux données
        shape, loc, scale = stats.pareto.fit(data)

        # Test de Kolmogorov-Smirnov pour évaluer l'ajustement de la distribution
        ks_statistic, ks_p_value = stats.kstest(data, 'pareto', (shape, loc, scale))

        # Calcul de la probabilité de poursuite pour une durée spécifique
        proba = stats.pareto.sf(last_length, shape, loc, scale)

        dict = {
            'Trend way' : way,
            'Last length' : "{:.2f}".format(last_length),
            'Last return' : "{:.2f}%".format(last_return*100),
            'Distribution type' : 'Pareto',
            'Trend description' : pd.Series(data).describe().round(2).to_dict(),
            'Distribution parameters' : {
                'Shape' : "{:.2f}".format(shape),
                'Loc' : "{:.2f}".format(loc),
                'Scale' : "{:.2f}".format(scale)
            },
            'KS test' : {
                'Statitics' : "{:.2f}%".format(ks_statistic*100),
                'P-value' : "{:.2f}%".format(ks_p_value*100)
            },
            'Continuation proba' : "{:.2f}%".format(proba*100)
        }

        [print(f'{x}: {y}') for x, y in dict.items()]

    def plot_trend(self, start, end, figsize=(12, 6), logy=False, show=True):
        history = self.history
        trend = self.trend
        ticker = self.ticker
        threshold = self.threshold
        timeperiod = self.timeperiod

        # Linéarisation de la tendance pour la représenter graphiquement
        linear_df = pd.DataFrame()
        for row in trend.index:
            linear_data = np.linspace(
                history.loc[trend.loc[row, 'First']],
                history.loc[trend.loc[row, 'Last']],
                len(history.loc[trend.loc[row, 'First'] : trend.loc[row, 'Last']])
            )
            index = history[trend.loc[row, 'First'] : trend.loc[row, 'Last']].index.to_list()
            linear_df = pd.concat([linear_df, pd.DataFrame(linear_data, index=index)])
        linear_df = linear_df.resample(timeperiod).last()

        # Clean linear trend
        linear_df = linear_df.drop_duplicates()
        linear_df = linear_df.rename(columns={0: 'Trend'})
        linear_df = pd.concat([history, linear_df], axis=1)

        # Filtrer les données pour la plage spécifiée
        linear_df = linear_df.loc[start:end]

        # Créer le graphique
        plt.figure(figsize=figsize)
        plt.plot(linear_df[ticker], label=ticker, color='blue', linewidth=1)
        plt.plot(linear_df['Trend'], label='Trend', color='red', linewidth=1)

        # Ajouter une échelle logarithmique si demandé
        if logy:
            plt.yscale('log')
            plt.title(f'{ticker} Price and Trend (Log Scale) {threshold:.2f}')
        else:
            plt.title(f'{ticker} Price and Trend {threshold:.2f}')

        # Ajouter une légende et une grille
        plt.legend()
        plt.grid(True)

        # Show the figure   
        if show:
            plt.show()
        else:
            trend_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(trend_file.name, format='png', bbox_inches='tight')
            plt.close()
            return trend_file.name

    def plot_distrib(self, figsize=(12, 6), show=True):
        history = self.history
        trend = self.trend
        ticker = history.name
        trend_data = trend['Length'].astype(float)
        last_trend = trend.iloc[-1]['Length']

        # Tracé de l'histogramme
        plt.figure(figsize=figsize)
        plt.hist(
            trend_data, 
            bins=30, 
            density=True, 
            alpha=0.7, 
            color='blue', 
            edgecolor='black', 
            label=ticker
        )

        # Tracé de la distribution de Pareto théorique
        shape, loc, scale = stats.pareto.fit(trend_data)
        x = np.linspace(trend_data.min(), trend_data.max(), 100)
        y = stats.pareto.pdf(x, shape, loc, scale)
        plt.plot(x, y, 'r-', lw=1, label='Pareto pdf')

        # Ajout de labels et d'une légende
        plt.ylabel('Frequency')
        plt.title(f'{ticker} - Distribution chart')

        plt.axvline(
            x=last_trend, 
            color='red', 
            linestyle='dashed', 
            linewidth=1, 
            label='Last trend'
        )

        plt.grid(True)
        plt.legend()

        # Show the figure   
        if show:
            plt.show()
        else:
            distrib_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(distrib_file.name, format='png', bbox_inches='tight')
            plt.close()
            return distrib_file.name

    def plot_cumdistrib(self, figsize=(12, 6), show=True):
        history = self.history
        trend = self.trend 
        ticker = history.name
        trend_data = trend['Length'].astype(float)

        # Calcul des caractéristiques de la séquence en cours 
        last_length = trend.iloc[-1]['Length']
        shape, loc, scale = stats.pareto.fit(trend_data)
        x = np.linspace(trend_data.min(), trend_data.max(), 100)
        y = stats.pareto.pdf(x, shape, loc, scale)
        
        # Plot cumulative distribution chart
        fig = plt.figure(figsize=figsize, layout="constrained")
        axs = fig.subplots(1, 2, sharex=True, sharey=True)

        y *= 10000 # Pas normal de devoir faire ça, besoin de chercher pourquoi
        y = y.cumsum()
        y /= y[-1]
        axs[0].ecdf(trend_data, label="CDF",color = 'blue')
        axs[0].plot(x, y, linewidth=1, label="Distribution pdf", color='red')
        axs[0].axvline(x=last_length, color='red', linestyle='dashed', linewidth=1, label='Last trend')

        # Complementary cumulative distributions
        axs[1].ecdf(trend_data, complementary=True, label="RCDF",color = 'blue')
        axs[1].plot(x, 1 - y, linewidth=1, label="Distribution pdf", color='red')
        axs[1].axvline(x=last_length, color='red', linestyle='dashed', linewidth=1, label='Last trend')

        # Label the figure.
        fig.suptitle(f"{ticker} - Cumulative Distribution Chart")
        for ax in axs:
            ax.grid(True)
            ax.legend()
            ax.set_xlabel("Length")
            ax.set_ylabel("Probability of occurrence")
            ax.label_outer()

        # Show the figure   
        if show:
            plt.show()
        else:
            cumdistrib_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(cumdistrib_file.name, format='png', bbox_inches='tight')
            plt.close()
            return cumdistrib_file.name
