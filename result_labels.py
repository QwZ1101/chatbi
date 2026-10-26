"""Display labels derived from stock_daily semantics, never browser translation."""
import re
import sqlglot
from sqlglot import exp

FIELDS = {
    'id': '记录编号', 'ts_code': '股票代码', 'stock_name': '股票名称',
    'trade_date': '交易日期', 'open': '开盘价（元）', 'high': '最高价（元）',
    'low': '最低价（元）', 'close': '收盘价（元）', 'pre_close': '前收盘价（元）',
    'change': '涨跌额（元）', 'pct_chg': '日涨跌幅（%）',
    'vol': '成交量（手）', 'amount': '成交额（千元）',
    'created_at': '记录创建时间', 'updated_at': '记录更新时间',
}


def expression_label(node):
    if isinstance(node, exp.Alias):
        node = node.this
    if isinstance(node, exp.Column):
        return FIELDS.get(node.name.lower())
    if isinstance(node, exp.Round):
        return expression_label(node.this)
    if isinstance(node, exp.Count):
        if isinstance(node.this, exp.Star):
            return '记录数'
        if isinstance(node.this, exp.Distinct):
            cols = node.this.expressions
            label = expression_label(cols[0]) if len(cols) == 1 else None
            return label + '去重数量' if label else None
        label = expression_label(node.this)
        return label + '非空数量' if label else None
    for kind, prefix in [(exp.Avg, '平均'), (exp.Sum, '合计'),
                         (exp.Min, '最小'), (exp.Max, '最大')]:
        if isinstance(node, kind):
            label = expression_label(node.this)
            if label == '交易日期' and kind in (exp.Min, exp.Max):
                return '最早交易日期' if kind == exp.Min else '最新交易日期'
            return prefix + label if label else None
    return None


def display_labels(columns, sql=''):
    labels = [FIELDS.get(str(c).lower(), str(c)) for c in columns]
    if sql:
        try:
            selections = sqlglot.parse_one(sql, read='mysql').expressions
            # Match by position, including duplicate aliases; SELECT * retains field labels.
            if len(selections) == len(columns) and not any(isinstance(e, exp.Star) for e in selections):
                labels = [expression_label(e) or str(c) for e, c in zip(selections, columns)]
        except (sqlglot.errors.ParseError, AttributeError):
            pass
    return labels


def result_table(frame, sql='', **kwargs):
    view = frame.copy()
    view.columns = display_labels(frame.columns, sql)
    html = view.to_html(index=False, border=0, justify='center', **kwargs)
    return html.replace('<table ', '<table translate="no" ', 1)


def ordered_metric(columns, sql):
    """Return a selected numeric ranking field without confusing it with a time series."""
    try:
        tree = sqlglot.parse_one(sql, read='mysql')
        order = tree.args.get('order')
        node = order.expressions[0].this if order else None
        if isinstance(node, exp.Column) and node.name in columns:
            return node.name
    except (sqlglot.errors.ParseError, AttributeError):
        pass
    return None


def metric_group(label):
    match = re.search(r'[（(]([^（）()]*)[）)]$', label)
    return match.group(1) if match else label
