# Clusterização de Clientes — RFM + KMeans

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=felipeeduardor/clusters-clientes&branch=master&mainModule=app.py)

App interativo para segmentação de clientes usando análise RFM e algoritmo KMeans.

## Como usar

1. Clique no botão **Open in Streamlit** acima
2. Faça upload do CSV de transações
3. Escolha o número de clusters e clique em **Processar**
4. Explore os gráficos e baixe o resultado

## Colunas obrigatórias no CSV

| Coluna | Descrição |
|---|---|
| `codcli` | Código do cliente |
| `data faturamento` | Data da transação |
| `quantidade_carregada` | Quantidade vendida |
| `valor_carregado` | Valor da venda |

## Tecnologias

- Python · Streamlit · Pandas · Scikit-learn · Plotly

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```
