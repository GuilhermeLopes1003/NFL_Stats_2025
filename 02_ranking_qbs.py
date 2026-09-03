import nflreadpy as nfl
import polars as pl

print("Carregando dados da temporada 2025...")
pbp = nfl.load_pbp([2025])

# Agrupar as jogadas por Quarterback e calcular as médias
ranking_qbs = (
    pbp
    # 1. Filtra apenas jogadas de passe com um lançador identificado
    .filter(
        (pl.col("play_type") == "pass") & 
        (pl.col("passer_player_name").is_not_null())
    )
    # 2. Agrupa pelo nome do QB e pelo time atacante
    .group_by(["passer_player_name", "posteam"])
    # 3. Calcula os totais e médias de cada estatística
    .agg([
        pl.len().alias("total_passes"),                    # Quantidade total de tentativas de passe
        pl.col("epa").mean().alias("epa_medio"),           # Média de eficiência (EPA por jogada)
        pl.col("cpoe").mean().alias("cpoe_medio"),         # Média de precisão de passe (%)
        pl.col("yards_gained").sum().alias("total_jardas") # Soma total de jardas lançadas
    ])
    # 4. Filtra apenas QBs titulares/com boa amostragem (mínimo de 100 passes)
    .filter(pl.col("total_passes") >= 100)
    # 5. Ordena do QB mais eficiente para o menos eficiente
    .sort("epa_medio", descending=True)
)

print("\n--- TOP 10 QUARTERBACKS DE 2025 (Por Eficiência - EPA/play) ---")
print(ranking_qbs.head(10))