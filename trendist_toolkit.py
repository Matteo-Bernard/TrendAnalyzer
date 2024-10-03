import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

class trends():
    def __init__(self, history, threshold, timeperiod = 'W'):
        self.history = history
        self.threshold = threshold
        self.timeperiod = timeperiod
        self.ticker = history.name

    def fit(self):
        history = self.history
        threshold = self.threshold
        timeperiod = self.timeperiod
        
        # Isolation de l'historique des prix ajustés de clôture pour l'actif étudié
        history = history.resample(timeperiod).last()
        history = history.dropna()

        # Initialisation du DataFrame pour stocker les caractéristiques des tendances
        trend = pd.DataFrame(columns=['Ticker', 'First', 'Last', 'Return', 'Length', 'Way'])
        d0 = history.index[0]

        # Boucle à travers chaque date dans l'historique pour calculer les tendances
        for d in history.index:
            p = history.loc[d]

            pmax = history.loc[d0:d].max()
            dmax = history.loc[history == pmax].index[0]
            pmin = history.loc[d0:d].min()
            dmin = history.loc[history == pmin].index[0]

            # Calcul du rendement en fonction du maximum ou du minimum précédent
            if dmin < dmax:
                r = (p - pmax) / pmax
            else:
                r = (p - pmin) / pmin

            # Filtrage des tendances en fonction du seuil
            if abs(r) > threshold:
                if dmin < dmax:
                    trend.at[d0, 'First'] = d0
                    trend.at[d0, 'Last'] = dmax
                    trend.at[d0, 'Return'] = r
                    trend.at[d0, 'Length'] = (dmax - d0).days
                    d0 = dmax
                        
                else:
                    trend.at[d0, 'First'] = d0
                    trend.at[d0, 'Last'] = dmin
                    trend.at[d0, 'Return'] = r
                    trend.at[d0, 'Length'] = (dmin - d0).days
                    d0 = dmin

        # Attribution des étiquettes '1' ou '0' en fonction du sens de la tendance
        for row in trend.index:
            if trend.loc[row, 'Return'] > 0:
                trend.at[row, 'Way'] = 1
            else:
                trend.at[row, 'Way'] = 0

        # Fusion des tendances adjacentes et de même sens pour avoir une atlernance entre tendances haussière et baissière
        prev_way = None
        prev_row = None
        for row in trend.index:
            way = trend.loc[row, 'Way']

            if prev_way == way:
                trend.at[prev_row, 'Last'] = trend.at[row, 'Last']
                trend = trend.drop(row)
            else:
                prev_row = row
                prev_way = way

        # Calcul ajusté de la durée et du rendement des tendances consolidées
        for row in trend.index:
            first = trend.loc[row, 'First']
            last = trend.loc[row, 'Last']
            trend.at[row, 'Return'] = (history[last] - history[first]) / history[first]
            trend.at[row, 'Length'] = (last - first).days
        trend = trend.reset_index(drop=True)

        # Attribution des étiquettes 'Up' ou 'Down' en fonction du sens de la tendance
        for row in trend.index:
            if trend.loc[row, 'Return'] > 0:
                trend.at[row, 'Way'] = 'Up'
            else:
                trend.at[row, 'Way'] = 'Down'
        self.trend = trend
        return self



    def get_proba(self, metric):
        history = self.history
        trend = self.trend
        # Calcul des caractéristiques de la séquence en cours 
        last_seq = history.loc[trend.iloc[-1]['Last']:]
        last_return = (last_seq.iloc[-1] - last_seq.iloc[0]) / last_seq.iloc[0]
        last_length = (history.index[-1] - trend.iloc[-1]['Last']).days

        # Extraction des données à partir du DataFrame
        if last_return > 0:
            data = trend[metric]
        else:
            data = trend[metric]
        data = data.to_list()

        # Ajustement des paramètres de la distribution de Pareto aux données
        shape, loc, scale = stats.pareto.fit(data)

        # Test de Kolmogorov-Smirnov pour évaluer l'ajustement de la distribution
        ks_statistic, ks_p_value = stats.kstest(data, 'pareto', (shape, loc, scale))

        # Calcul de la probabilité de poursuite pour une durée spécifique
        proba = stats.pareto.sf(last_length, shape, loc, scale)

        dict = {
            'Last length' : "{:.2f}".format(last_length),
            'Last return' : "{:.2f}%".format(last_return),
            'Trend description' : pd.Series(data).describe().to_dict(),
            'Distribution parameters' : {
                'Shape' : shape,
                'Loc' : loc,
                'Scale' : scale
            },
            'KS test' : {
                'Statitics' : ks_statistic,
                'P-value' : ks_p_value
            },
            'Continuation proba' : proba
        }
        return dict
    


    def get_trend(self):
        """
        Renvoi le DataFrame de tendances
        """
        return self.trend



    def plot_trend(self):
        history = self.history
        trend = self.trend
        timeperiod = self.timeperiod
        ticker = self.ticker
        threshold = self.threshold

        # Linéarisation de la tendance pour la représenter graphiquement
        history = history.resample('D').ffill()
        linear_df = pd.DataFrame()
        for row in trend.index:
            linear_data = np.linspace(
                history.loc[trend.loc[row, 'First']],
                history.loc[trend.loc[row, 'Last']],
                len(history.loc[trend.loc[row, 'First'] : trend.loc[row, 'Last']])
            )
            index = history[trend.loc[row, 'First'] : trend.loc[row, 'Last']].index.to_list()
            linear_df = pd.concat([linear_df, pd.DataFrame(linear_data, index = index)])
        history = history.resample(timeperiod).last()

        # Clean linear trend 
        linear_df = linear_df.drop_duplicates()
        linear_df = linear_df.resample(timeperiod).last()
        linear_df = linear_df.rename(columns={0:'Trend'})
        linear_df = pd.concat([history, linear_df], axis=1)

        linear_df = linear_df.loc[history.index[-1] - pd.Timedelta(weeks=52*5):]
        plt.figure(figsize=(8, 4))
        plt.plot(linear_df[ticker], label=ticker, color='blue', linewidth=1)
        plt.plot(linear_df['Trend'], label='Trend', color='red', linewidth=1)

        # Adding title and legend
        plt.title(f'{ticker} Price and Trend {threshold:.2f}')
        plt.legend()
        #plt.grid(True)
        plt.show()



    def plot_distrib(self, metric):
        history = self.history
        trend = self.trend
        ticker = history.name
        trend_data = trend[metric].astype(float)

        # Calcul des caractéristiques de la séquence en cours 
        last_seq = history.loc[trend.iloc[-1]['Last']:]
        last_length = (history.index[-1] - trend.iloc[-1]['Last']).days

        # Tracé de l'histogramme
        plt.figure(figsize=(8, 4))
        plt.hist(
            trend_data, 
            bins=25, 
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
        plt.plot(
            x, 
            y, 
            'r-', 
            lw=1, 
            label='Pareto pdf'
        )

        # Ajout de labels et d'une légende
        plt.ylabel('Frequency')
        plt.title(f'{ticker} - Distribution chart')
        plt.axvline(
            x=last_length, 
            color='red', 
            linestyle='dashed', 
            linewidth=1, 
            label='Last trend'
        )

        #plt.grid(True)
        plt.legend()
        plt.show()



    def plot_cumdistrib(self, metric):
        history = self.history
        trend = self.trend 
        ticker = history.name
        trend_data = trend[metric].astype(float)

        # Calcul des caractéristiques de la séquence en cours 
        last_seq = history.loc[trend.iloc[-1]['Last']:]
        last_return = (last_seq.iloc[-1] - last_seq.iloc[0]) / last_seq.iloc[0]
        last_length = (history.index[-1] - trend.iloc[-1]['Last']).days
        
        if metric == 'Length':
            last_metric = last_length
        elif metric == 'Return':
            last_metric = last_return

        shape, loc, scale = stats.pareto.fit(trend_data)
        x = np.linspace(trend_data.min(), trend_data.max(), 100)
        y = stats.pareto.pdf(x, shape, loc, scale)
        
        # Plot cumulative distribution chart
        fig = plt.figure(figsize=(12, 4), layout="constrained")
        axs = fig.subplots(1, 2, sharex=True, sharey=True)

        y *= 10000 # Pas normal de devoir faire ça, besoin de rechercher
        y = y.cumsum()
        y /= y[-1]
        axs[0].ecdf(
            trend_data, 
            label="CDF",
            color = 'blue'
        )
        axs[0].plot(
            x, 
            y, 
            linewidth=1, 
            label="Pareto pdf", 
            color='red'
        )
        axs[0].axvline(
            x=last_metric, 
            color='red', 
            linestyle='dashed', 
            linewidth=1, 
            label='Last trend'
        )

        # Complementary cumulative distributions
        axs[1].ecdf(
            trend_data, 
            complementary=True, 
            label="RCDF",
            color = 'blue'
        )
        axs[1].plot(
            x, 
            1 - y, 
            linewidth=1, 
            label="Pareto pdf", 
            color='red'
        )
        axs[1].axvline(
            x=last_metric, 
            color='red', 
            linestyle='dashed', 
            linewidth=1, 
            label='Last trend'
        )

        # Label the figure.
        fig.suptitle(f"{ticker} - Cumulative Distribution Chart")
        for ax in axs:
            ax.grid(True)
            ax.legend()
            ax.set_xlabel("Length")
            ax.set_ylabel("Probability of occurrence")
            ax.label_outer()
        plt.show()