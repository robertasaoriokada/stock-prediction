import time

from .library import pd, yf, sgs
from .stock_config import START, END, SERIES_BCB, DATA_FOLDER


def import_sgs(name, code, start=START, end=END):
    """O SGS não permite mais de 10 anos, então vamos usar 5 anos para garantir que não haja problemas."""
    chunks = []
    start_ts = pd.Timestamp(start)
    fim_ts = pd.Timestamp(end)
    while start_ts <= fim_ts:
        f = min(start_ts + pd.DateOffset(years=5), fim_ts)
        chunks.append(
            sgs.get(
                codes={name: code},
                start=start_ts.strftime("%Y-%m-%d"),
                end=f.strftime("%Y-%m-%d"),
            )
        )
        start_ts = f + pd.Timedelta(days=1)
    s = pd.concat(chunks)
    s.index = pd.to_datetime(s.index)
    return s[~s.index.duplicated()]


def _download(ticker, start, end, retries=3):
    """Baixa um ticker com Ticker.history e devolve colunas simples (Open, High, Low, Close, Volume)."""
    raw = pd.DataFrame()
    for tentativa in range(1, retries + 1):
        try:
            raw = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=True,  # explícito: Close ajustado por dividendos/splits
                actions=False,     # não traz as colunas Dividends/Stock Splits
            )
        except Exception as e:
            print(f"[{ticker}] tentativa {tentativa} falhou: {e}")
            raw = pd.DataFrame()
        if not raw.empty:
            break
        time.sleep(2 * tentativa)

    if raw.empty:
        raise RuntimeError(f"Yahoo não retornou dados para {ticker}.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # remove o fuso e mantém a data local do pregão (evita deslocar o dia)
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    return raw


def closing(ticker, name, start, end):
    raw = _download(ticker, start, end)
    return pd.DataFrame(
        {"date": raw.index.astype("datetime64[ns]"), name: raw["Close"].values}
    )


def building_database_stock(ticker, prefix, start=START, end=END):
    """Monta o dataset diário: ativo (OHLCV) + Ibovespa + VIX + séries do BCB."""
    raw = _download(ticker, start, end)
    df = pd.DataFrame(
        {
            "date": raw.index.astype("datetime64[ns]"),
            f"{prefix}_close": raw["Close"].values,
            f"{prefix}_high": raw["High"].values,
            f"{prefix}_low": raw["Low"].values,
            f"{prefix}_open": raw["Open"].values,
            f"{prefix}_volume": raw["Volume"].values,
        }
    )

    for t, nome in [("^BVSP", "ibovespa"), ("^VIX", "vix")]:
        df = df.merge(closing(t, nome, start, end), on="date", how="left")

    # merge_asof: cada dia recebe o último valor conhecido da série (inclusive o IPCA mensal)
    df = df.sort_values("date")
    for nome, codigo in SERIES_BCB.items():
        s = import_sgs(nome, codigo, start, end).rename_axis("date").reset_index()
        s["date"] = pd.to_datetime(s["date"]).astype("datetime64[ns]")
        df = pd.merge_asof(df, s.sort_values("date"), on="date")

    return df.ffill().reset_index(drop=True)


def load_stock(ticker, prefix, force=False):
    """Lê de data/<prefix>.csv; só baixa se o arquivo não existir ou se forcar=True."""
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    caminho = DATA_FOLDER / f"{prefix}.csv"
    if caminho.exists() and not force:
        return pd.read_csv(caminho, parse_dates=["date"])
    df = building_database_stock(ticker, prefix)
    df.to_csv(caminho, index=False)
    return df