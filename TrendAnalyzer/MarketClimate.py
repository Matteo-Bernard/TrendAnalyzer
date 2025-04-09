import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tempfile

def MC(df, x_value, y_value, show=True):

    plt.figure(figsize=(10, 10))

    sns.scatterplot(
        x=x_value,
        y=y_value, 
        hue=df.index.year, 
        palette='plasma', 
        data=df, 
        legend='full',
        s=15,
        alpha=0.75
    )

    plt.title('Risk and Reward climate indicator')

    plt.axvline(x=df[x_value].mean(), color='black', linewidth=1)
    plt.hlines(y=df[y_value].mean(), xmin=-1000000000, xmax=1000000000, color='black', linewidth=1)

    # Définir les limites des axes x et y pour centrer le graphique
    x_margin = np.percentile(df[x_value], 10) - np.percentile(df[x_value], 90)
    y_margin = np.percentile(df[y_value], 10) - np.percentile(df[y_value], 90)

    plt.xlim(df[x_value].mean() + x_margin,df[x_value].mean() - x_margin)
    plt.ylim(df[y_value].mean() + y_margin,df[y_value].mean() - y_margin)
    
    # Tracer le dernier point de la série en rouge
    plt.scatter(
        x = df[x_value].iloc[-1],
        y = df[y_value].iloc[-1], 
        color='black',
        s=20
    )

    # Ajouter une légende à côté du point rouge
    plt.annotate(
        df[y_value].index[-1].strftime('%d-%m-%Y'), 
        xy=(df[x_value].iloc[-1], df[y_value].iloc[-1]),
        xytext=(5, -5),
        textcoords='offset points', 
        color='black', 
        fontsize=10,
    )

    plt.legend(title='Year', bbox_to_anchor=(1, 1), loc='upper left')

    if show:
        plt.show()
    else:
        mc_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(mc_file.name, format='png', bbox_inches='tight')
        plt.close()
        return mc_file
