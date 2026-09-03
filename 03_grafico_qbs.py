import nflreadpy as nfl
import polars as pl
import plotly.express as px

print("Carregando e processando dados...")
pbp = nfl.load_pbp([2025])

# 1. Mesma agregação que você acabou de rodar
ranking_qbs = (
    pbp
    .filter(
        (pl.col("play_type") == "pass") & 
        (pl.col("passer_player_name").is_not_null())
    )
    .group_by(["passer_player_name", "posteam"])
    .agg([
        pl.len().alias("total_passes"),
        pl.col("epa").mean().alias("epa_medio"),
        pl.col("cpoe").mean().alias("cpoe_medio"),
        pl.col("yards_gained").sum().alias("total_jardas")
    ])
    .filter(pl.col("total_passes") >= 100)
)

# 2. Converter de Polars para Pandas apenas para o Plotly ler com facilidade
df_plot = ranking_qbs.to_pandas()

# 3. Criar o Gráfico com Plotly
fig = px.scatter(
    df_plot,
    x="cpoe_medio",
    y="epa_medio",
    text="passer_player_name",      # Rótulo com o nome do jogador no ponto
    size="total_passes",            # Tamanho do ponto pelo volume de passes
    color="epa_medio",              # Cor mudando conforme a eficiência
    color_continuous_scale="RdYlGn", # Escala de cor: Vermelho (baixo) a Verde (alto)
    title="NFL 2025: Eficiência (EPA/play) vs. Precisão (CPOE) dos Quarterbacks",
    labels={
        "cpoe_medio": "Precisão (%) - CPOE Médio",
        "epa_medio": "Eficiência - EPA Médio por Jogada"
    }
)

# Ajusta o texto do nome do jogador para ficar acima da bolinha
fig.update_traces(textposition="top center")

# Desenha linhas tracejadas com a média da liga para criar os 4 quadrantes
fig.add_hline(y=df_plot["epa_medio"].mean(), line_dash="dash", line_color="gray")
fig.add_vline(x=df_plot["cpoe_medio"].mean(), line_dash="dash", line_color="gray")

# 4. Abre o gráfico interativo direto no seu navegador
fig.show()