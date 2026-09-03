import nflreadpy as nfl

# Descobre qual é a temporada atual
temporada = nfl.get_current_season()
print(f"Conexão OK! Temporada identificada: {temporada}")