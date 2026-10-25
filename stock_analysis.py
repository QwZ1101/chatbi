"""Deterministic analytics over bounded, parameterized read-only queries."""
from dataclasses import dataclass
from datetime import date
import re

import numpy as np
import pandas as pd

from db_access import read_dataframe
from sql_guard import MAX_ROWS


@dataclass
class AnalysisResult:
    table: pd.DataFrame
    chart: pd.DataFrame
    note: str


def validate_request(args):
    if not isinstance(args, dict):
        raise ValueError('参数必须为对象')
    mode = args.get('mode')
    if mode not in {'interval_return', 'monthly_close', 'daily_stats'}:
        raise ValueError('mode 只支持 interval_return、monthly_close、daily_stats')
    codes = args.get('ts_codes')
    if (not isinstance(codes, list) or not 1 <= len(codes) <= 4 or
            any(not isinstance(c, str) or not re.fullmatch(r'[0-9]{6}\.(SH|SZ|BJ)', c) for c in codes)):
        raise ValueError('ts_codes 必须包含 1～4 个完整股票代码，例如 600519.SH')
    codes = list(dict.fromkeys(codes))
    dates = []
    for key in ('start_date', 'end_date'):
        raw = args.get(key)
        if not isinstance(raw, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', raw):
            raise ValueError(key + ' 必须为 YYYY-MM-DD')
        dates.append(date.fromisoformat(raw))
    if dates[0] > dates[1] or (dates[1] - dates[0]).days > 3660:
        raise ValueError('开始日期不得晚于结束日期，查询范围不得超过 3660 天')
    return mode, codes, dates[0], dates[1]


def analyze_stocks(args):
    mode, codes, start, end = validate_request(args)
    frames = []
    missing = []
    # One bounded SELECT per stock; no model-supplied SQL or interpolated values.
    for code in codes:
        frame = read_dataframe(
            'SELECT ts_code, stock_name, trade_date, close, pct_chg FROM stock_daily '
            'WHERE ts_code = :code AND trade_date BETWEEN :start AND :end '
            'ORDER BY trade_date ASC',
            params={'code': code, 'start': start.isoformat(), 'end': end.isoformat()})
        if len(frame) >= MAX_ROWS:
            raise ValueError('查询达到行数上限，无法确认数据完整；请缩短区间后重试')
        if frame.empty:
            missing.append(code)
            continue
        frame = frame.copy()
        frame['trade_date'] = pd.to_datetime(frame['trade_date'], errors='raise')
        if (frame['trade_date'].isna().any() or frame.duplicated(['ts_code', 'trade_date']).any()
                or not frame['ts_code'].eq(code).all()
                or not frame['trade_date'].dt.date.between(start, end).all()):
            raise ValueError('行情包含重复日期、错误股票代码或越界日期，停止计算')
        column = 'pct_chg' if mode == 'daily_stats' else 'close'
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(column + ' 存在空值或非有限数值，停止计算，避免遗漏端点或改变分母')
        if column == 'close' and frame[column].le(0).any():
            raise ValueError('收盘价必须大于 0，停止计算')
        frames.append(frame.sort_values('trade_date'))
    if not frames:
        return AnalysisResult(pd.DataFrame(), pd.DataFrame(), '所选股票在指定区间没有行情记录。')
    data = pd.concat(frames, ignore_index=True)
    note = (f'请求区间：{start} 至 {end}。按库内实际观测计算，不补齐缺失交易日；'
            '结果中的日期为实际取值日期，不保证数据覆盖完整。')
    if missing:
        note += ' 未查到数据的股票：' + '、'.join(missing) + '。'
    if mode == 'monthly_close':
        data['月份'] = data['trade_date'].dt.strftime('%Y-%m')
        last = data.groupby(['ts_code', '月份'], sort=True).tail(1).sort_values(['月份', 'ts_code'])
        table = last[['月份', 'ts_code', 'stock_name', 'trade_date', 'close']].copy()
        table.columns = ['月份', '股票代码', '股票名称', '实际取值日期', '月内末条收盘价(元)']
        table['实际取值日期'] = table['实际取值日期'].dt.strftime('%Y-%m-%d')
        chart = last.pivot(index='月份', columns='ts_code', values='close').reset_index()
        note += ' 月末口径为所选区间内每月最后一条行情，非当月最高价；未结束月份或缺行情月份仅表示截至可用日期。'
        return AnalysisResult(table, chart, note)
    rows = []
    for code, frame in data.groupby('ts_code', sort=True):
        first, last = frame.iloc[0], frame.iloc[-1]
        common = {'股票代码': code, '股票名称': str(last['stock_name']),
                  '首个日期': first['trade_date'].strftime('%Y-%m-%d'),
                  '末个日期': last['trade_date'].strftime('%Y-%m-%d'), '观测天数': len(frame)}
        if mode == 'interval_return':
            common.update({'首日收盘价(元)': float(first['close']), '末日收盘价(元)': float(last['close']),
                           '区间涨跌幅(%)': round((float(last['close']) / float(first['close']) - 1) * 100, 4)
                           if len(frame) >= 2 else None,
                           '状态': '正常' if len(frame) >= 2 else '仅一条行情，无法计算区间涨跌幅'})
        else:
            returns = frame['pct_chg']
            common.update({'上涨天数': int(returns.gt(0).sum()), '下跌天数': int(returns.lt(0).sum()),
                           '平盘天数': int(returns.eq(0).sum()), '上涨日占比(%)': round(returns.gt(0).mean() * 100, 4)})
        rows.append(common)
    table = pd.DataFrame(rows)
    metric = '区间涨跌幅(%)' if mode == 'interval_return' else '上涨日占比(%)'
    table = table.sort_values(metric, ascending=False, na_position='last')
    chart = table[['股票代码', metric]].copy()
    note += (' 区间涨跌幅=(末日收盘价/首日收盘价-1)×100%，使用库内收盘价，不额外做复权或总回报调整。'
             if mode == 'interval_return' else ' 上涨日占比=上涨天数/全部有效观测天数，分母包含平盘日；不代表未来获利概率。')
    return AnalysisResult(table, chart, note)
