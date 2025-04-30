import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

def get_dataframe_summary():
    df = pd.read_csv("sales_data_sample.csv", encoding='unicode_escape')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df = pd.get_dummies(df.drop(columns=['STATUS']), columns=['COUNTRY', 'PRODUCTLINE', 'DEALSIZE'])
    df['cluster'] = KMeans(3).fit_predict(PCA(2).fit_transform(StandardScaler().fit_transform(df.select_dtypes('number'))))
    return df[['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'cluster']]

def generate_plots():
    df = pd.read_csv("sales_data_sample.csv", encoding='unicode_escape')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

    # Sales por país
    df['COUNTRY'].value_counts().plot(kind='bar', figsize=(10, 5), color='cornflowerblue')
    plt.title("Ventas por País")
    plt.tight_layout()
    plt.savefig('app/static/plots/sales_by_country.png')
    plt.close()

    # Matriz de correlación
    df = pd.get_dummies(df, columns=['COUNTRY', 'PRODUCTLINE', 'DEALSIZE'])
    corr = df.select_dtypes('number').corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title("Matriz de Correlación")
    plt.savefig("app/static/plots/mi_grafico.png")
    plt.close() 

    # Clustering 2D
    scaled = StandardScaler().fit_transform(df.select_dtypes(include='number'))
    pca = PCA(n_components=2).fit_transform(scaled)
    labels = KMeans(n_clusters=3).fit_predict(pca)
    fig = px.scatter(x=pca[:, 0], y=pca[:, 1], color=labels.astype(str), title="Clustering 2D")
    fig.write_image("app/static/plots/clusters_2d.png")

    # Histogramas (simplificado)
    df['cluster'] = labels
    df_numeric = df.select_dtypes(include='number')
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
    for ax, col in zip(axes.flatten(), df_numeric.columns[:8]):
        for cluster in df['cluster'].unique():
            df[df['cluster'] == cluster][col].plot.hist(alpha=0.5, ax=ax, label=f'Cluster {cluster}')
        ax.set_title(col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("app/static/plots/histograms_cluster.png")
    plt.close()

def generate_3d_plot():
    df = pd.read_csv("sales_data_sample.csv", encoding='unicode_escape')

    # Preprocesamiento
    df = pd.get_dummies(df.drop(columns=['STATUS']), columns=['COUNTRY', 'PRODUCTLINE', 'DEALSIZE'])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.select_dtypes(include='number'))

    # PCA + Clustering
    pca = PCA(n_components=3)
    components = pca.fit_transform(scaled)
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(components)

    # DataFrame con resultados
    pca_df = pd.DataFrame(components, columns=['pca1', 'pca2', 'pca3'])
    pca_df['cluster'] = labels.astype(str)

    # Gráfico 3D
    fig = px.scatter_3d(
        pca_df, x='pca1', y='pca2', z='pca3',
        color='cluster',
        title='Clustering 3D',
        opacity=0.7,
        symbol='cluster',
        width=800,
        height=600
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
