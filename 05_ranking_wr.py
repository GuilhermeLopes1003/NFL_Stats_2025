import nflreadpy as nfl
import polars as pl

print("Carregando dados da temporada 2025...")
pbp = nfl.load_pbp([2025])

print("Filtrando a lista oficial de Wide Receivers...")
rosters = nfl.load_rosters([2025])

# Criamos a lista de IDs únicos apenas de quem é 'WR'
ids_wrs = (
    rosters
    .filter(pl.col("position") == "WR")
    .select("gsis_id")
    .to_series()
    .to_list()
)

ranking_wrs = (
    pbp
        .filter(
            (pl.col("play_type") == "pass") &
            (pl.col("receiver_player_name").is_not_null()) &
            pl.col("receiver_player_id").is_in(ids_wrs)
            )

        .group_by(["receiver_player_name", "posteam"])
        .agg([
                pl.len().alias("alvos"),
                pl.col("complete_pass").sum().alias("recepcoes"),
                pl.col("complete_pass").mean().alias("taxa_captura"),
                pl.col("yards_gained").filter(pl.col("complete_pass") == 1).sum().alias("jardas"),
                pl.col("pass_touchdown").filter(pl.col("complete_pass") == 1).sum().alias("tds"),
                pl.col("epa").mean().alias("epa_medio")
            ])
        .filter(pl.col("alvos") >= 100)
        .with_columns(
            (pl.col("jardas") / pl.col("alvos")).alias("jardas_por_alvo")
            )
        .sort("jardas_por_alvo", descending=True)

        )
print("\n--- TOP 10 Wide Receivers DE 2025 (Por Eficiência - Jardas por Alvo ---")
print(ranking_wrs.head(10))




