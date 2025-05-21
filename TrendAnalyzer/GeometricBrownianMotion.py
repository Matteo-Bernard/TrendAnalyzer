import numpy as np
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

class GBM:
    def __init__(self, 
                 n_years=10,
                 n_scenarios=1000,
                 mu=0.07,
                 sigma=0.15,
                 steps_per_year=12,
                 s_0=100.0,
                 prices=True):
        # Initialiser les paramètres du modèle GBM
        self.n_years = n_years                      # Durée de la simulation (en années)
        self.n_scenarios = n_scenarios              # Nombre de trajectoires simulées
        self.mu = mu                                # Rendement espéré annualisé
        self.sigma = sigma                          # Volatilité annualisée
        self.steps_per_year = steps_per_year        # Nombre de pas par an (granularité)
        self.s_0 = s_0                               # Valeur initiale
        self.prices = prices                        # Si True, renvoie les prix ; sinon, les rendements

    def generate_path(self):
        # Générer les trajectoires GBM sous forme de DataFrame
        dt = 1 / self.steps_per_year                                # Pas de temps
        n_steps = int(self.n_years * self.steps_per_year) + 1       # Nombre total de pas
        # Générer les rendements log-normaux : E[ret+1] = (1 + mu)^dt, écart-type = sigma * sqrt(dt)
        rets_plus_1 = np.random.normal(
            loc=(1 + self.mu) ** dt,
            scale=self.sigma * np.sqrt(dt),
            size=(n_steps, self.n_scenarios)
        )
        rets_plus_1[0] = 1  # Premier point à 1 pour assurer un point de départ commun

        if self.prices:
            # Retourner les prix simulés par produit cumulatif
            return self.s_0 * pd.DataFrame(rets_plus_1).cumprod()
        else:
            # Retourner les rendements (rets_plus_1 - 1)
            return pd.DataFrame(rets_plus_1 - 1)

    def plot_path(self):
        # Afficher les trajectoires simulées
        path = self.generate_path()
        ax = path.plot(
            legend=False, 
            color="indianred", 
            alpha=0.5, 
            linewidth=2, 
            figsize=(12, 6)
        )
        # Ajouter une ligne horizontale représentant la valeur initiale
        ax.axhline(y=self.s_0, ls=":", color="black")
        # Marquer le point de départ
        ax.plot(0, self.s_0, marker='o', color='darkred', alpha=0.2)
        plt.show()

    def display_path(self):
        # Créer une interface interactive avec ipywidgets pour ajuster les paramètres
        def _update(n_scenario, returns, volatility, capital, n_years):
            # Mettre à jour les paramètres de la simulation
            self.n_scenarios = n_scenario
            self.mu = returns
            self.sigma = volatility
            self.s_0 = capital
            self.n_years = n_years
            self.plot_path()

        # Création des widgets interactifs
        controls = widgets.interactive(
            _update,
            n_scenario=widgets.IntSlider(min=1, max=1000, step=1, value=self.n_scenarios),
            returns=widgets.FloatSlider(min=0., max=0.2, step=0.01, value=self.mu),
            volatility=widgets.FloatSlider(min=0., max=0.3, step=0.01, value=self.sigma),
            capital=widgets.IntSlider(min=100, max=10000, step=100, value=self.s_0),
            n_years=widgets.IntSlider(min=1, max=25, step=1, value=self.n_years)
        )
        # Afficher l'interface
        display(controls)