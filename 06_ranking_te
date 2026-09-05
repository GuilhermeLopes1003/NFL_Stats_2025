import nflreadpy as nfl
import polars as pl

print("Carregando dados da temporada 2025...")
pbp = nfl.load_pbp([2025])

print("Filtrando a lista oficial de Tight Ends...")
rosters = nfl.load_rosters([2025])

# Lista de IDs de quem é 'TE'
ids_tes = (
    rosters
    .filter(pl.col("position") == "TE")
    .select("gsis_id")
    .to_series()
    .to_list()
)

ranking_tes = (
    pbp
    .filter(
        (pl.col("play_type") == "pass") &
        (pl.col("receiver_player_name").is_not_null()) &
        pl.col("receiver_player_id").is_in(ids_tes)
    )
    .group_by(["receiver_player_name", "posteam"])
    .agg([
        pl.len().alias("alvos"),
        pl.col("complete_pass").sum().cast(pl.UInt32).alias("recepcoes"),
        (pl.col("complete_pass").mean() * 100).alias("taxa_captura"),
        # Casts adicionados para limpar os decimais .0
        pl.col("yards_gained").filter(pl.col("complete_pass") == 1).sum().cast(pl.Int64).alias("jardas"),
        pl.col("pass_touchdown").filter(pl.col("complete_pass") == 1).sum().cast(pl.UInt32).alias("tds"),
        pl.col("epa").mean().alias("epa_medio")
    ])
    .filter(pl.col("alvos") >= 40)  # Corte ideal de volume para TEs
    .with_columns(
        (pl.col("jardas") / pl.col("alvos")).alias("jardas_por_alvo")
    )
    .sort(["jardas_por_alvo", "jardas"], descending=[True, True])
)

print("\n--- TOP 10 Tight Ends DE 2025 (Por Eficiência - Jardas por Alvo) ---")
print(ranking_tes.head(10))