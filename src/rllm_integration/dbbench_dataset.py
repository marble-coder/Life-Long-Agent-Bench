import json
import os
from typing import Any, Dict, List

from rllm.data.dataset import Dataset, DatasetRegistry


def _load_raw_standard(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Raw file is {idx: sample_dict}; convert to ordered list and keep sample_index
    rows: List[Dict[str, Any]] = []
    for key, sample in sorted(raw.items(), key=lambda kv: int(kv[0])):
        sample = dict(sample)
        sample["sample_index"] = key
        rows.append(sample)
    return rows


def register_dbbench_standard(name: str = "dbbench_standard_local", data_path: str = "data/db_bench.json") -> Dataset:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"DBBench data file not found: {data_path}")
    data = _load_raw_standard(data_path)
    # 避免 parquet 转换嵌套结构报错，直接返回内存 Dataset，不写注册表
    return Dataset(data=data, name=name, split="train")


def load_dbbench_standard(name: str = "dbbench_standard_local", data_path: str = "data/db_bench.json") -> Dataset:
    return register_dbbench_standard(name=name, data_path=data_path)
