# -*- coding: utf-8 -*-
"""
使用 tushare 获取指定股票的历史价格（日线），保存到 Excel。

股票：
  - 贵州茅台  600519.SH
  - 五粮液    000858.SZ
  - 国泰君安  601211.SH
  - 中芯国际  688981.SH

区间：2025-01-01 ~ 今天
输出：stock_history.xlsx （所有股票合并写入单个“汇总” Sheet）

运行前需设置环境变量 TUSHARE_TOKEN：
  PowerShell:  $env:TUSHARE_TOKEN = "你的token"
  Linux/macOS: export TUSHARE_TOKEN="你的token"
"""

import datetime as dt
import os

import pandas as pd
import tushare as ts

# ---------- 配置 ----------
TOKEN_ENV_VAR = "TUSHARE_TOKEN"

STOCKS = {
    "贵州茅台": "600519.SH",
    "五粮液": "000858.SZ",
    "国泰君安": "601211.SH",
    "中芯国际": "688981.SH",
}

START_DATE = "20250101"
END_DATE = dt.date.today().strftime("%Y%m%d")
OUTPUT_FILE = "stock_history.xlsx"

# 需要保留的字段（英文列名，与 1.sql 中 stock_daily 表字段一致）
COLUMNS_MAP = {
    "trade_date": "trade_date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "pre_close": "pre_close",
    "change": "change",
    "pct_chg": "pct_chg",
    "vol": "vol",
    "amount": "amount",
}


def fetch_one(pro, name: str, code: str) -> pd.DataFrame:
    """获取单只股票的日线行情。"""
    print(f"正在获取 {name} ({code}) : {START_DATE} ~ {END_DATE} ...")
    df = pro.daily(ts_code=code, start_date=START_DATE, end_date=END_DATE)
    if df is None or df.empty:
        print(f"  ! {name} 未获取到数据")
        return pd.DataFrame()

    # 只保留存在的目标列，并按日期升序排列
    keep = [c for c in COLUMNS_MAP if c in df.columns]
    df = df[keep].copy()
    df = df.sort_values("trade_date").reset_index(drop=True)
    df = df.rename(columns=COLUMNS_MAP)
    # trade_date 格式化为 YYYY-MM-DD
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    print(f"  获取 {len(df)} 条记录")
    return df


def load_token() -> str:
    """从环境变量读取 tushare token，缺失时给出清晰报错。"""
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        raise SystemExit(
            f"错误：未检测到环境变量 {TOKEN_ENV_VAR}。请先设置你的 tushare token 再运行本脚本：\n"
            f"  PowerShell:  $env:{TOKEN_ENV_VAR} = \"你的token\"\n"
            f"  Linux/macOS: export {TOKEN_ENV_VAR}=\"你的token\"\n"
            f"（token 可在 https://tushare.pro 个人主页获取）"
        )
    return token


def main():
    token = load_token()
    ts.set_token(token)
    pro = ts.pro_api(token)

    all_rows = []
    for name, code in STOCKS.items():
        df = fetch_one(pro, name, code)
        if df.empty:
            continue
        tagged = df.copy()
        tagged.insert(0, "stock_name", name)
        tagged.insert(1, "ts_code", code)
        all_rows.append(tagged)

    if not all_rows:
        print("\n未获取到任何数据，不生成文件。")
        return

    summary = pd.concat(all_rows, ignore_index=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="汇总", index=False)

    print(f"\n完成！已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
