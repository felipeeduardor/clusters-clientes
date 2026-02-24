import streamlit as st
import pandas as pd
import plotly.express as px

from rfm_engine import calcular_rfm, rodar_kmeans, calcular_inercia

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clusterização de Clientes",
    page_icon="🎯",
    layout="wide",
)

st.title("Clusterização de Clientes — RFM + KMeans")
st.markdown("Faça upload do CSV de transações para calcular os segmentos automaticamente.")

# ---------------------------------------------------------------------------
# Seção 1: Upload e Preview
# ---------------------------------------------------------------------------
st.header("1. Upload dos Dados")

uploaded_file = st.file_uploader(
    "Selecione o arquivo CSV de clientes",
    type=["csv"],
    help="Colunas obrigatórias: codcli, data faturamento, quantidade_carregada, valor_carregado",
)

if uploaded_file is None:
    st.info("Aguardando upload do arquivo CSV...")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

# Validação de colunas obrigatórias
REQUIRED_COLS = {"codcli", "data faturamento", "quantidade_carregada", "valor_carregado"}
missing_cols = REQUIRED_COLS - set(df_raw.columns)
if missing_cols:
    st.error(f"Colunas obrigatórias ausentes no arquivo: {missing_cols}")
    st.stop()

# Métricas rápidas
df_dates = pd.to_datetime(df_raw["data faturamento"], errors="coerce")
periodo = (
    f"{df_dates.min().strftime('%d/%m/%Y')} → {df_dates.max().strftime('%d/%m/%Y')}"
    if df_dates.notna().any()
    else "—"
)

col1, col2, col3 = st.columns(3)
col1.metric("Total de Linhas", f"{len(df_raw):,}")
col2.metric("Clientes Únicos", f"{df_raw['codcli'].nunique():,}")
col3.metric("Período", periodo)

st.dataframe(df_raw.head(10), width='stretch')

# ---------------------------------------------------------------------------
# Seção 2: Configuração e Processamento
# ---------------------------------------------------------------------------
st.header("2. Configuração e Processamento")

n_clusters = st.slider(
    "Número de Clusters (k)",
    min_value=2,
    max_value=10,
    value=4,
    help="Use o gráfico Elbow abaixo para escolher o k ideal após processar.",
)

processar = st.button("Processar", type="primary")

if processar:
    with st.spinner("Calculando RFM e rodando KMeans... aguarde."):
        df_rfm = calcular_rfm(df_raw)
        inertia = calcular_inercia(df_rfm)
        df_result = rodar_kmeans(df_rfm, n_clusters)

    st.session_state["df_result"] = df_result
    st.session_state["inertia"] = inertia
    st.session_state["n_clusters"] = n_clusters
    st.success(f"Processamento concluído! {df_result['Cluster'].nunique()} clusters gerados.")

# Exibe gráfico Elbow se disponível
if "inertia" in st.session_state:
    inertia = st.session_state["inertia"]
    k_sel = st.session_state.get("n_clusters", 4)

    fig_elbow = px.line(
        x=list(range(1, len(inertia) + 1)),
        y=inertia,
        markers=True,
        title="Método Elbow — Escolha do Número Ideal de Clusters",
        labels={"x": "Número de Clusters (k)", "y": "Inércia"},
    )
    fig_elbow.add_vline(
        x=k_sel,
        line_dash="dash",
        line_color="red",
        annotation_text=f"k={k_sel} selecionado",
        annotation_position="top right",
    )
    st.plotly_chart(fig_elbow, width='stretch')

# ---------------------------------------------------------------------------
# Seções 3 e 4 — exibidas somente após processamento
# ---------------------------------------------------------------------------
if "df_result" not in st.session_state:
    st.stop()

df_result = st.session_state["df_result"]

# Converter Cluster para string para cores discretas nos gráficos
df_plot = df_result.copy()
df_plot["Cluster"] = df_plot["Cluster"].astype(str)

# ---------------------------------------------------------------------------
# Seção 3: Visualizações dos Clusters
# ---------------------------------------------------------------------------
st.header("3. Visualizações dos Clusters")

# --- Scatter: Recency x Frequency (tamanho = Monetary, cor = Cluster) ---
st.subheader("Dispersão: Recency × Frequency")
fig_scatter = px.scatter(
    df_plot,
    x="Recency",
    y="Frequency",
    size="Monetary",
    color="Cluster",
    hover_name="codcli",
    hover_data=["RFM_Score", "Monetary"],
    size_max=60,
    title="Distribuição dos Clientes por Cluster",
    labels={"Recency": "Recency (dias)", "Frequency": "Frequency (qtd total)"},
    color_discrete_sequence=px.colors.qualitative.Bold,
)
st.plotly_chart(fig_scatter, width='stretch')

# --- Boxplots R, F, M por Cluster ---
st.subheader("Distribuição de R, F, M por Cluster")
col_r, col_f, col_m = st.columns(3)

with col_r:
    fig_r = px.box(
        df_plot, x="Cluster", y="Recency",
        color="Cluster",
        title="Recency por Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_r.update_layout(showlegend=False)
    st.plotly_chart(fig_r, width='stretch')

with col_f:
    fig_f = px.box(
        df_plot, x="Cluster", y="Frequency",
        color="Cluster",
        title="Frequency por Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_f.update_layout(showlegend=False)
    st.plotly_chart(fig_f, width='stretch')

with col_m:
    fig_m = px.box(
        df_plot, x="Cluster", y="Monetary",
        color="Cluster",
        title="Monetary por Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_m.update_layout(showlegend=False)
    st.plotly_chart(fig_m, width='stretch')

# --- Barras: Monetary médio e total por Cluster ---
st.subheader("Valor Monetário por Cluster")
col_avg, col_sum = st.columns(2)

with col_avg:
    df_avg = df_plot.groupby("Cluster", as_index=False)["Monetary"].mean()
    fig_avg = px.bar(
        df_avg, x="Cluster", y="Monetary",
        color="Cluster",
        title="Valor Monetário Médio por Cluster",
        text_auto=".3s",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_avg.update_layout(showlegend=False)
    st.plotly_chart(fig_avg, width='stretch')

with col_sum:
    df_sum = df_plot.groupby("Cluster", as_index=False)["Monetary"].sum()
    fig_sum = px.bar(
        df_sum, x="Cluster", y="Monetary",
        color="Cluster",
        title="Valor Monetário Total por Cluster",
        text_auto=".3s",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_sum.update_layout(showlegend=False)
    st.plotly_chart(fig_sum, width='stretch')

# --- Tabela dinâmica RFM_Score × Cluster ---
st.subheader("Tabela Dinâmica: RFM_Score × Cluster")
pivot = df_result.pivot_table(
    index="RFM_Score",
    columns="Cluster",
    values="codcli",
    aggfunc="count",
    fill_value=0,
)
pivot.columns = [f"Cluster {c}" for c in pivot.columns]
st.dataframe(pivot, width='stretch')

# ---------------------------------------------------------------------------
# Seção 4: Download dos Resultados
# ---------------------------------------------------------------------------
st.header("4. Resultados e Download")

COLS_EXPORT = ["codcli", "data faturamento", "Recency", "Frequency", "Monetary", "RFM_Score", "Cluster"]
st.dataframe(df_result[COLS_EXPORT], width='stretch')

csv_bytes = df_result[COLS_EXPORT].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Baixar CSV com Clusters",
    data=csv_bytes,
    file_name="clientes_clusters.csv",
    mime="text/csv",
)
