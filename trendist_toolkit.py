import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

class trends():
    def __init__(self, history, threshold, timeperiod = 'W'):
        history = history.resample(timeperiod).last()
        self.history = history
        self.threshold = threshold
        self.ticker = history.name
        self.timeperiod = timeperiod

    def fit(self):
        history = self.history
        threshold = self.threshold

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
            trend.at[row, 'Ticker'] = history.name
        trend = trend.reset_index(drop=True)

        # Attribution des étiquettes 'Up' ou 'Down' en fonction du sens de la tendance
        for row in trend.index:
            if trend.loc[row, 'Return'] > 0:
                trend.at[row, 'Way'] = 'Up'
            else:
                trend.at[row, 'Way'] = 'Down'

        # Calculer la tendance en cours
        last_trend = pd.DataFrame()
        first = trend.iloc[-1]['Last']
        last = history.index[-1]
        last_trend.at[row+1, 'Ticker'] = history.name
        last_trend.at[row+1, 'First'] = first
        last_trend.at[row+1, 'Last'] = last
        last_trend.at[row+1, 'Return'] = ((history[last] - history[first]) / history[first])
        last_trend.at[row+1, 'Length'] = (last - first).days
        last_trend.at[row+1, 'Way'] = 'Up' if last_trend.loc[row+1, 'Return'] > 0 else 'Down'
        trend = pd.concat([trend, last_trend])

        #trend.loc[trend['Length'] < 0, 'Length'] = 100 # à corriger

        self.trend = trend
        return self



    def get_proba(self, metric):
        history = self.history
        trend = self.trend

        # Calcul des caractéristiques de la séquence en cours 
        last_length = trend.iloc[-1]['Length']
        last_return = trend.iloc[-1]['Return']
        way = trend.iloc[-1]['Way']
        data = trend[metric].to_list()

        if metric == 'Length':

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
                    'Shape' : round(shape,2),
                    'Loc' : round(loc,2),
                    'Scale' : round(scale,2)
                },
                'KS test' : {
                    'Statitics' : "{:.2f}%".format(ks_statistic*100),
                    'P-value' : "{:.2f}%".format(ks_p_value*100)
                },
                'Continuation proba' : "{:.2f}%".format(proba*100)
            }
        
        elif metric == 'Return':

            # Ajustement des paramètres de la distribution de Pareto aux données
            mean, std = stats.norm.fit(data)

            # Test de Kolmogorov-Smirnov pour évaluer l'ajustement de la distribution
            ks_statistic, ks_p_value = stats.kstest(data, 'norm', (mean, std))

            # Calcul de la probabilité de poursuite pour une durée spécifique
            proba = stats.norm.sf(last_return, mean, std)
    
            dict = {
                'Trend way' : way,         
                'Last length' : "{:.2f}".format(last_length),
                'Last return' : "{:.2f}%".format(last_return*100),
                'Distribution type' : 'Normal',
                'Trend description' : pd.Series(data).describe().round(2).to_dict(),
                'Distribution parameters' : {
                    'Mean' : "{:.2f}%".format(mean*100),
                    'Std' : "{:.2f}%".format(std*100),
                },
                'KS test' : {
                    'Statitics' : "{:.2f}%".format(ks_statistic*100),
                    'P-value' : "{:.2f}%".format(ks_p_value*100)
                },
                'Continuation proba' : "{:.2f}%".format(proba*100)
            }

        [print(f'{x}: {y}') for x, y in dict.items()]
    


    def get_trend(self):
        """
        Renvoi le DataFrame de tendances
        """
        return self.trend



    def plot_trend(self):
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
            linear_df = pd.concat([linear_df, pd.DataFrame(linear_data, index = index)])
        linear_df = linear_df.resample(timeperiod).last()

        # Clean linear trend 
        linear_df = linear_df.drop_duplicates()
        linear_df = linear_df.rename(columns={0:'Trend'})
        linear_df = pd.concat([history, linear_df], axis=1)

        linear_df = linear_df.loc[history.index[-1] - pd.Timedelta(weeks=52*5):]
        plt.figure(figsize=(8, 4))
        plt.plot(linear_df[ticker], label=ticker, color='blue', linewidth=1)
        plt.plot(linear_df['Trend'], label='Trend', color='red', linewidth=1)

        # Adding title and legend
        plt.title(f'{ticker} Price and Trend {threshold:.2f}')
        plt.legend()
        plt.grid(True)
        plt.show()



    def plot_distrib(self, metric):
        history = self.history
        trend = self.trend
        ticker = history.name
        trend_data = trend[metric].astype(float)

        # Calcul des caractéristiques de la séquence en cours 
        last_seq = history.loc[trend.iloc[-1]['Last']:]
        last_length = (history.index[-1] - trend.iloc[-1]['Last']).days
        
        last_trend = trend.iloc[-1][metric]

        # Tracé de l'histogramme
        plt.figure(figsize=(8, 4))
        plt.hist(
            trend_data, 
            bins=30, 
            density=True, 
            alpha=0.7, 
            color='blue', 
            edgecolor='black', 
            label=ticker
        )

        if metric == 'Length':

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
        
        elif metric == 'Return':

            # Tracé de la distribution normale théorique
            mean, std = stats.norm.fit(trend_data)
            x = np.linspace(trend_data.min(), trend_data.max(), 100)
            y = stats.norm.pdf(x, mean, std)
            plt.plot(
                x, 
                y, 
                'r-', 
                lw=1, 
                label='Normal pdf'
            )

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
        plt.show()



    def plot_cumdistrib(self, metric):
        history = self.history
        trend = self.trend 
        ticker = history.name
        trend_data = trend[metric].astype(float)

        # Calcul des caractéristiques de la séquence en cours 
        last_length = trend.iloc[-1]['Length']
        last_return = trend.iloc[-1]['Return']
        
        if metric == 'Length':
            last_metric = last_length
        elif metric == 'Return':
            last_metric = last_return

        if metric == 'Length':
            shape, loc, scale = stats.pareto.fit(trend_data)
            x = np.linspace(trend_data.min(), trend_data.max(), 100)
            y = stats.pareto.pdf(x, shape, loc, scale)
        elif metric == 'Return':
            mean, std = stats.norm.fit(trend_data)
            x = np.linspace(trend_data.min(), trend_data.max(), 100)
            y = stats.norm.pdf(x, mean, std)
        
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
            label="Distribution pdf", 
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
            label="Distribution pdf", 
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