import nflreadpy as nfl
import polars as pl

print("Carregando dados da temporada 2025...")
pbp = nfl.load_pbp([2025])

print("Filtrando a lista oficial de Running Backs...")
rosters = nfl.load_rosters([2025])

# Criamos a lista de IDs únicos apenas de quem é 'RB'
ids_rbs = (
    rosters
    .filter(pl.col("position") == "RB")
    .select("gsis_id")
    .to_series()
    .to_list()
)

# A sua estrutura exata que você já domina:
ranking_rbs = (
    pbp
    .filter(
        (pl.col("play_type") == "run") & 
        (pl.col("rusher_player_name").is_not_null()) &
        (pl.col("rusher_player_id").is_in(ids_rbs))  # <--- TRAVA ANTI-QB
    )
    
    .group_by(["rusher_player_name", "posteam"])
    # 3. Calcula os totais e médias de cada estatística
    .agg([
        pl.len().alias("total_corridas"),
        pl.col("epa").gt(0).mean().alias("taxa_sucesso"),
        pl.col("epa").mean().alias("epa_medio"),           # Média de eficiência (EPA por jogada)
        pl.col("rush_touchdown").sum().alias("total_tds"),
        pl.col("yards_gained").sum().alias("total_jardas") # Soma total de jardas lançadas
    ])
    .filter(pl.col("total_corridas") >= 150)
    # 5. Ordena do QB mais eficiente para o menos eficiente
    .sort("taxa_sucesso", descending=True)
)

print("\n--- TOP 10 RUNNING BACKS DE 2025 (Por Eficiência - Taxa de sucesso ---")
print(ranking_rbs.head(10))