import pandas as pd
import json
import os
import ast

def parse_field(value):
    """将字符串形式的字典/列表解析为真正的对象"""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except:
            return json.loads(value)
    return value

# 读取 parquet 文件
df = pd.read_parquet(r'data/os_data.parquet')

# 创建输出目录
output_dir = 'data/v0303/os_interaction/processed/v0409_tcc_9_to_12_first500'
os.makedirs(output_dir, exist_ok=True)

# 转换为 entry_dict 格式
entry_dict = {}
for idx, row in df.iterrows():
    entry = {
        "instruction": row["instruction"],
        "initialization_command_item": parse_field(row["initialization_command_item"]),
        "evaluation_info": parse_field(row["evaluation_info"]),
        "skill_list": parse_field(row["skill_list"]),
        "raw_entry_hash": int(row["raw_entry_hash"])
    }
    entry_dict[str(row["sample_index"])] = entry

# 保存为 JSON
output_path = os.path.join(output_dir, 'entry_dict.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(entry_dict, f, indent=2, ensure_ascii=False)

print(f"转换完成！")
print(f"输出路径: {output_path}")
print(f"样本数量: {len(entry_dict)}")

# 验证第一条数据
print("\n=== 验证第一条数据类型 ===")
first_entry = entry_dict["0"]
print(f"initialization_command_item 类型: {type(first_entry['initialization_command_item'])}")
print(f"evaluation_info 类型: {type(first_entry['evaluation_info'])}")
print(f"skill_list 类型: {type(first_entry['skill_list'])}")
