# 一次性把 stock_history.xlsx 的「汇总」Sheet 导入 MySQL chatbi.stock_daily
# 用法：source venv/bin/activate && set -a && source .env && set +a && python import_xlsx.py
import os

import pandas as pd
from sqlalchemy import create_engine, text

DB_USER = os.getenv("MYSQL_USER", "chatbi")
DB_PASS = os.getenv("MYSQL_PASSWORD", "")
DB_HOST = os.getenv("MYSQL_HOST", "localhost")
DB_PORT = os.getenv("MYSQL_PORT", "3306")
DB_NAME = os.getenv("MYSQL_DB", "chatbi")

COLS = ["stock_name", "ts_code", "trade_date", "open", "high", "low", "close",
        "pre_close", "change", "pct_chg", "vol", "amount"]


def main():
    df = pd.read_excel("stock_history.xlsx", sheet_name="汇总")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"「汇总」Sheet 缺少列: {missing}，实际列: {list(df.columns)}")
    df = df[COLS].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    engine = create_engine(
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4")
    with engine.begin() as conn:
        conn.execute(text("SET SESSION sql_mode=''"))
        df.to_sql("stock_daily", con=conn, if_exists="append", index=False)
    print(f"导入完成：{len(df)} 行 -> {DB_NAME}.stock_daily")


if __name__ == "__main__":
    main()
