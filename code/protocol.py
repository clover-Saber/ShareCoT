"""Nine-feature protocol for the main comparison."""
import hashlib
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
SERIALIZATION_VERSION = 'nine-feature-semantic-dictionary-usd-v2'
_SEMANTIC_DICTIONARY = {}


def configure_semantic_dictionary(data_directory=None):
    """Load the training-defined categorical vocabulary without fitting on data.

    Readable categorical values can already be supplied in CSVs. Encoded indices
    require an explicit dictionary; their meanings are never inferred or guessed.
    The same vocabulary must accompany train, validation and test records.
    """
    global _SEMANTIC_DICTIONARY
    path = None if data_directory is None else Path(data_directory) / 'semantic_dictionary.json'
    dictionary = json.loads(path.read_text(encoding='utf-8')) if path is not None and path.is_file() else {}
    allowed = {'trip_start_timestamp', 'dropoff_poi_category'}
    if not isinstance(dictionary, dict) or set(dictionary) - allowed:
        raise ValueError('semantic_dictionary.json must contain only timestamp and POI mappings')
    for field, mapping in dictionary.items():
        if not isinstance(mapping, dict) or any(
                not isinstance(label, str) or not label.strip() for label in mapping.values()):
            raise ValueError(f'semantic_dictionary.json: {field} requires nonempty text labels')
    _SEMANTIC_DICTIONARY = dictionary
    return {'serialization_version': SERIALIZATION_VERSION,
            'semantic_dictionary_sha256': sha256(path) if path is not None and path.is_file() else None}


def semantic_label(field, value):
    """Apply V[index] from manuscript Eq. (12), preserving decoded text values."""
    text = str(value).strip()
    mapping = _SEMANTIC_DICTIONARY.get(field, {})
    if text in mapping:
        return mapping[text]
    # CSV readers can promote an integer column with missing cells to floats.
    if re.fullmatch(r'[+-]?\d+\.0+', text) and str(int(text.split('.')[0])) in mapping:
        return mapping[str(int(text.split('.')[0]))]
    if not text or text.lower() in ('nan', 'none', '<na>'):
        return '未提供'
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text):
        raise ValueError(f'{field}={text} needs an entry in data/semantic_dictionary.json')
    return text


def feature_text(row):
    missing = set(FEATURE_COLS) - set(row)
    if missing:
        raise ValueError(f'Missing allowed features: {sorted(missing)}')
    return (
        '请判断本次行程是否授权拼车。\n'
        f"- 出发时间：{semantic_label('trip_start_timestamp', row['trip_start_timestamp'])}\n"
        f"- 下车 POI 类别：{semantic_label('dropoff_poi_category', row['dropoff_poi_category'])}\n"
        f"- 行程：{row['trip_seconds']} 秒，{row['trip_miles']} 英里\n"
        f"- 费用：车费={row['fare']} USD，附加费用={row['additional_charges']} USD\n"
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


def feature_fingerprint(row):
    data = json.dumps({k: str(row[k]) for k in FEATURE_COLS}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)




