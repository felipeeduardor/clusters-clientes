import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def calcular_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Recency, Frequency, Monetary por codcli.
    Retorna df_rfm com as 3 métricas e RFM_Score.
    """
    df = df.copy()
    df['data faturamento'] = pd.to_datetime(df['data faturamento'])

    # Data de referência: data mais recente no dataset
    data_atual = df['data faturamento'].max()

    # Recency: dias desde a última compra
    recency = df.groupby('codcli')['data faturamento'].max().reset_index()
    recency['Recency'] = (data_atual - recency['data faturamento']).dt.days

    # Frequency: soma total de quantidade_carregada
    frequency = df.groupby('codcli')['quantidade_carregada'].sum().reset_index()
    frequency.rename(columns={'quantidade_carregada': 'Frequency'}, inplace=True)

    # Monetary: soma total de valor_carregado
    monetary = df.groupby('codcli')['valor_carregado'].sum().reset_index()
    monetary.rename(columns={'valor_carregado': 'Monetary'}, inplace=True)

    # Merge das métricas
    df_rfm = recency.merge(frequency, on='codcli').merge(monetary, on='codcli')

    # Quartis RFM (Recency: pontuação inversa — menor Recency = melhor)
    df_rfm['R_Quartile'] = pd.qcut(
        df_rfm['Recency'], 4, labels=range(4, 0, -1), duplicates='drop'
    )
    df_rfm['F_Quartile'] = pd.qcut(
        df_rfm['Frequency'], 4, labels=range(1, 5), duplicates='drop'
    )
    df_rfm['M_Quartile'] = pd.qcut(
        df_rfm['Monetary'], 4, labels=range(1, 5), duplicates='drop'
    )

    df_rfm['RFM_Score'] = (
        df_rfm['R_Quartile'].astype(str)
        + df_rfm['F_Quartile'].astype(str)
        + df_rfm['M_Quartile'].astype(str)
    )

    return df_rfm


def rodar_kmeans(df_rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    StandardScaler + KMeans.
    Retorna df_rfm com coluna 'Cluster'.
    """
    df_rfm = df_rfm.copy()

    X = df_rfm[['Recency', 'Frequency', 'Monetary']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_rfm['Cluster'] = kmeans.fit_predict(X_scaled)

    return df_rfm


def calcular_inercia(df_rfm: pd.DataFrame, k_max: int = 10) -> list:
    """
    Elbow method — retorna lista de inércias para k=1..k_max.
    """
    X = df_rfm[['Recency', 'Frequency', 'Monetary']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia = []
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    return inertia
