from pathlib import Path
 
# Pasta raiz do projeto (C:\TCC) e pasta onde ficam os CSVs
ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = ROOT / "data"
 
# Período e divisão treino/teste
START = "2012-03-01"
END = "2025-12-31"
TRAIN_FRAC = 0.8
SEED = 42
 
# prefixo das colunas -> ticker no Yahoo Finance
STOCKS = {
    "vale": "VALE3.SA",
    "petro": "PETR4.SA",
    "itub": "ITUB4.SA",
}
 
# Séries do BCB (SGS)
SERIES_BCB = {
    "selic_daily": 11,
    "selic_goal": 432,
    "ipca": 433,
    "fx_usd": 1,
    "gdp": 4380,
    "unemployment": 24369,
}
 
