import nflreadpy as nfl
import polars as pl

print("Baixando dados da temporada 2025...")
# 1. Baixar o Play-by-Play da temporada de 2025
pbp_bruto = nfl.load_pbp([2025])

print(f"Total de jogadas registradas: {len(pbp_bruto)}")

# 2. Filtrar apenas jogadas de passe válidas e selecionar colunas corretas
passes = (
    pbp_bruto
    .filter(
        (pl.col("play_type") == "pass") & 
        (pl.col("passer_player_name").is_not_null())
    )
    .select([
        "passer_player_name",  # Nome do Quarterback que fez o lançamento na jogada
        "posteam",             # Time que está no ataque com a posse da bola (Possession Team)
        "epa",                 # Expected Points Added: quanto a jogada aumentou a expectativa de pontos do time
        "cpoe",                # Completion Percentage Over Expected: precisão do passe acima/abaixo do esperado (%)
        "yards_gained"         # Total de jardas reais avançadas (ou perdidas) na jogada
    ])
)

print(f"Total de passes válidos: {len(passes)}")

# 3. Mostrar as 5 primeiras linhas
print("\nPrimeiras 5 jogadas de passe:")
print(passes.head(5))