import json
import re
from pathlib import Path

FEATURE_COLS = [
    'fare', 'trip_miles', 'trip_seconds', 'additional_charges',
    'pickup_to_center_distance_km', 'dropoff_to_center_distance_km',
    'trip_start_timestamp', 'dropoff_poi_category', 'weather_dewpoint_2m',
]
TARGET = 'shared_trip_authorized'
SYSTEM = '你正在执行行程级拼车授权二分类任务。请依据给定的行程信息预测是否授权拼车。'
FORBIDDEN = re.compile(r'小费|打赏|价格敏感|体验敏感|tip_paid|tip_propensity|\btips?\b|\btipp(?:ed|ing)\b|gratuity|price.sensitiv|experience.sensitiv', re.I)


def feature_text(row):
    missing = set(FEATURE_COLS) - set(row)
    if missing:
        raise ValueError(f'Missing allowed features: {sorted(missing)}')
    return (
        '请判断本次行程是否授权拼车。\n'
        f"- 出发时间：{row['trip_start_timestamp']}\n"
        f"- 下车 POI 类别：{row['dropoff_poi_category']}\n"
        f"- 行程：{row['trip_seconds']} 秒，{row['trip_miles']} 英里\n"
        f"- 费用：车费={row['fare']}，附加费用={row['additional_charges']}\n"
        f"- 距市中心距离：上车点={row['pickup_to_center_distance_km']} km，下车点={row['dropoff_to_center_distance_km']} km\n"
        f"- 天气露点：{row['weather_dewpoint_2m']}°C\n"
    )


def build_prompt(row, tokenizer):
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM},
         {'role': 'user', 'content': feature_text(row)}],
        tokenize=False, add_generation_prompt=True)


def rationale_is_clean(text):
    return isinstance(text, str) and bool(text.strip()) and not FORBIDDEN.search(text)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def verify_identity(actual, expected):
    if actual != expected:
        raise ValueError('Experiment provenance differs; use a new directory rather than mixing results')
