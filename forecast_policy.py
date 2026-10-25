"""Keep forecast steps anchored to their actual last observation."""
from datetime import date


def forecast_origin(last_date, today=None, max_age_days=7):
    today = today or date.today()
    if last_date > today:
        raise ValueError('行情末日晚于今天，日期异常，暂停预测')
    if (today - last_date).days > max_age_days:
        raise ValueError(f'行情截至 {last_date}，距今天超过 {max_age_days} 个自然日；请更新数据后再预测')
    return last_date
