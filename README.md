# run experiment
# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4,5  
export PYTHONPATH=./
# 运行实验
python ./src/run_experiment.py --config_path "local_usc16_config.yaml"

# SFT版本修改lora配置
在configs/components/callbacks/test_time_training_callback.yaml路径下去修改相关参数

# 清理docker容器
#!/bin/bash
echo "🧹 清理所有 MySQL Docker 容器..."

echo "停止容器..."
docker stop $(docker ps -q --filter ancestor=mysql) 2>/dev/null

echo "删除容器..."
docker rm -f $(docker ps -aq --filter ancestor=mysql) 2>/dev/null

echo "✅ 清理完成！当前 MySQL 容器："
docker ps -a | grep mysql || echo "无"

# 运行llama3-8b的memory+召回实验
export DASHSCOPE_API_KEY=sk-umjcynnsnunlyumgcdhuzdeorkldwwgsyfjokuvyogeskqnk
python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/trajectory_memory_usc16.yaml"

# 运行test-time-training只训练assistant部分实验
python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/sft_assistant_only.yaml"

# 运行test-time-training只训练assistant部分+ 利用历史4条成功轨迹实验
python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/previous_sample_utilization_usc4.yaml"

# 运行test-time-training只训练assistant部分+ memory召回
python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/sft_onlyassistant_memory.yaml"

# 运行qwen2.5-7b的memory策略
export DASHSCOPE_API_KEY=sk-30949268f306427886e6613da83a9e08
python ./src/run_experiment.py --config_path "configs/assignments/experiments/qwen25_7b_instruct/instance/db_bench/instance/memory.yaml"

# 运行qwen2.5-7b standard策略
python ./src/run_experiment.py --config_path "configs/assignments/experiments/qwen25_7b_instruct/instance/db_bench/instance/standard.yaml"

# 运行memory实验时要设置API KEY调用qwen-plus
export DASHSCOPE_API_KEY=sk-30949268f306427886e6613da83a9e08


# 修改轨迹策略 改为用embedding model召回相关轨迹(4条)
python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/previous_sample_embedding_usc4.yaml"

# 运行正常的轨迹添加上下文策略
python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/previous_sample_utilization_usc1.yaml"

# 运行embedding model召回相关轨迹+ttt增强
python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/previous_sample_embedding_ttt_clean_usc4.yaml"

# 运行反思记忆
export DASHSCOPE_API_KEY=sk-3c7d8138a66943ba9643ccebda724a00
python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/baseline_reflective_memory.yaml"
# LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners

<p align="center">
    <img src="https://img.picui.cn/free/2025/05/21/682d857c0cb55.png" alt="Logo" width="80px">

[//]: # (    <br>)
[//]: # (    <b>WebArena is a standalone, self-hostable web environment for building autonomous agents</b>)
</p>

<p align="center">
<a href="https://www.python.org/downloads/release/python-3119/"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"></a>
<a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
<a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/mypy-strict-blue" alt="Checked with mypy"></a>
</p>

<p align="center">
<a href="https://caixd-220529.github.io/LifelongAgentBench/">ProjectPage</a> •
<a href="https://arxiv.org/abs/2505.11942">Paper</a> •
<a href="https://huggingface.co/datasets/csyq/LifelongAgentBench">Dataset</a>
</p>

# Setup

```shell
git clone ...
cd continual_agent_bench
pip install -r requirements.txt
pip install pre-commit==4.0.1  # ensure that pre-commit hooks are installed
pre-commit install  # install pre-commit hooks
pre-commit run --all-files  # check its effect

docker pull mysql  # build images for db_bench

docker pull ubuntu  # build images for os_interaction
docker build -f scripts/dockerfile/os_interaction/default scripts/dockerfile/os_interaction --tag local-os/default
```

# Run experiments
If you want to run experiments in single machine mode, please use the following command:
```shell
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml"
```

If you want to run experiments in distributed mode, you first need to start the `ServerSideController` in the machine that can deploy the docker containers.
```shell
export PYTHONPATH=./

python src/distributed_deployment_utils/server_side_controller/main.py
```
Then, you can run the following command in HPC node.
```shell
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/distributed_deployment_utils/run_experiment_remotely.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml"
```
The `ServerSideController` can be reused for multiple experiments.
> [!NOTE]
> Don't forget to update the IP address in `configs/components/environment.yaml` as well as in the files under `configs/components/clients`.