import argparse
import json
import os
import yaml
import copy
from enum import StrEnum
from typing import Any, Mapping, Sequence, Optional
import coredumpy  # type: ignore[import-untyped]

from src.utils import ConfigLoader, SingletonLogger
from src.typings import (
    AssignmentConfig,
    EnvironmentConfig,
    SampleStatus,
    LoggerConfig,
    ContinualAgentBenchException,
    Session,
    SampleIndex,
    PathConfig,
    GeneralInstanceFactory,
    SessionMetricCalculationPartial,
    SessionEvaluationOutcome,
)
from src.tasks import Task, DatasetItem
from src.agents import Agent
from src.language_models import LanguageModel
from src.callbacks import (
    CallbackHandler,
    CallbackConstructor,
    Callback,
    CallbackRestorer,
    CallbackArguments,
)
try:
    from src.callbacks.instance import GRPOTrainingCallback
except Exception:
    GRPOTrainingCallback = None  # type: ignore
try:
    from src.callbacks.instance import GRPOTrainingCallbackRLLM
except Exception:
    GRPOTrainingCallbackRLLM = None  # type: ignore
try:
    from src.callbacks.instance import MemoryReviewGRPOCallback
except Exception:
    MemoryReviewGRPOCallback = None  # type: ignore
try:
    from src.callbacks.instance import RerankerGRPOTrainingCallback
except Exception:
    RerankerGRPOTrainingCallback = None  # type: ignore
try:
    from src.callbacks.instance import ReflectionGRPOTrainingCallback
except Exception:
    ReflectionGRPOTrainingCallback = None  # type: ignore


class ConfigUtilityCaller(StrEnum):
    CLIENT = "client"
    SERVER = "server"
    CLIENT_SIDE_CONTROLLER = "client_side_controller"


class ConfigUtility:
    def __init__(
        self,
        assignment_config: AssignmentConfig,
        environment_config: EnvironmentConfig,
        path_config: PathConfig,
    ):
        self.assignment_config = assignment_config
        self.environment_config = environment_config
        self.path_config = path_config

    def preprocess(self) -> None:
        if self.environment_config.task_client:
            self.assignment_config.task = self.environment_config.task_client

    def construct(self) -> tuple[Task[DatasetItem], Agent, dict[str, Callback]]:
        # Maybe task will be Task or TaskClient, but it doesn't matter!
        task: Task[DatasetItem] = self.assignment_config.task.create()
        # region Construct language_model_dict
        # Here, We actually instantiate the language models.
        # After the code exit construct(), We will never get a chance to get the language model instance.
        # But I think this is good, since this improve the modularity and maintainability of the code.
        language_model_dict: Mapping[str, LanguageModel] = {
            key: value.create()
            for key, value in self.assignment_config.language_model_dict.items()
        }
        agent_instance_factory: GeneralInstanceFactory = self.assignment_config.agent
        if (
            language_model_name := agent_instance_factory.parameters.get(
                "language_model"
            )
        ) is not None:
            agent_instance_factory.parameters["language_model"] = language_model_dict[
                language_model_name
            ]
        # endregion
        agent: Agent = agent_instance_factory.create()
        callback_dict = CallbackConstructor.construct(
            self.assignment_config, task, agent, language_model_dict
        )
        return task, agent, callback_dict

    def validate(self, task: Task[DatasetItem], agent: Agent) -> None:
        sample_index_list = task.get_sample_index_list()
        for selected_sample_index in self.assignment_config.sample_order:
            assert selected_sample_index in sample_index_list

    def postprocess(self, task: Task[DatasetItem], agent: Agent) -> None:
        if self.assignment_config.sample_order == "default":
            self.assignment_config.sample_order = task.get_sample_index_list()

    def remove_redundant_args(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        # Maybe use `if raw_config["environment_config"]["use_task_client_flag"]` is better, but I use the following
        # condition avoid using dict key directly.
        if not self.environment_config.task_client:
            # If the config file is used to restore the previous incomplete assignment, the `task_client` will be None.
            # Using del to remove the key-value pair will cause an error in this case.
            for key in list(raw_config["environment_config"]):
                if key != "use_task_client_flag":
                    del raw_config["environment_config"][key]
        redundant_key_buffer: set[tuple[str, str]] = set()
        for key in raw_config["task_dict"]:
            if key != raw_config["assignment_config"]["task"]:
                redundant_key_buffer.add(("task_dict", key))
        for key in raw_config["agent_dict"]:
            if key != raw_config["assignment_config"]["agent"]["name"]:
                redundant_key_buffer.add(("agent_dict", key))
        assignment_language_model_name_list: Sequence[str] = [
            language_model_info_dict["name"]
            for language_model_info_dict in raw_config["assignment_config"][
                "language_model_list"
            ]
        ]
        for key in raw_config["language_model_dict"]:
            if key not in assignment_language_model_name_list:
                redundant_key_buffer.add(("language_model_dict", key))
        assignment_callback_name_list: Sequence[str] = [
            callback_info_dict["name"]
            for callback_info_dict in raw_config["assignment_config"][
                "callback_dict"
            ].values()
        ]
        for key in raw_config["callback_dict"]:
            if key not in assignment_callback_name_list:
                redundant_key_buffer.add(("callback_dict", key))
        for info_tuple in redundant_key_buffer:
            del raw_config[info_tuple[0]][info_tuple[1]]
        return raw_config

    @staticmethod
    def _get_custom_instance_info_dict(
        default_instance_info_dict: Mapping[str, Any],
        custom_instance_info_dict: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        module: str = default_instance_info_dict["module"]
        # `... or {}` is used to ensure that both default_parameters and custom_parameters are dict.
        default_parameters: dict[str, Any] = copy.deepcopy(
            default_instance_info_dict.get("parameters") or {}
        )
        custom_parameters = custom_instance_info_dict.get("custom_parameters") or {}
        for parameter_name, custom_parameter_value in custom_parameters.items():
            if parameter_name not in default_parameters:
                raise AssertionError(
                    f"Parameter '{parameter_name}' not found in default parameters for module '{module}'. "
                    f"Available parameters: {list(default_parameters.keys())}"
                )
            # Overwrite the default parameter value with the custom parameter value.
            default_parameters[parameter_name] = custom_parameter_value
        return {
            "module": module,
            "parameters": default_parameters,
        }

    @staticmethod
    def read_raw_config(
        raw_config: Mapping[str, Any], caller: ConfigUtilityCaller
    ) -> tuple[AssignmentConfig, EnvironmentConfig, LoggerConfig, PathConfig]:
        raw_config = copy.deepcopy(raw_config)  # Avoid modifying the original config.
        # region Convert raw_config into assignment_config
        # region Construct assignment_language_model_dict
        assignment_language_model_list: Sequence[Mapping[str, Any]] = raw_config[
            "assignment_config"
        ]["language_model_list"]
        assignment_language_model_dict: dict[str, Any] = {}
        for language_model_info_dict in assignment_language_model_list:
            language_model_name = language_model_info_dict["name"]
            default_language_model_info_dict = raw_config["language_model_dict"][
                language_model_name
            ]
            assert language_model_name not in assignment_language_model_dict
            assignment_language_model_dict[language_model_name] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_language_model_info_dict, language_model_info_dict
                )
            )
        # endregion
        # region Construct assignment_agent
        custom_agent_info_dict = raw_config["assignment_config"]["agent"]
        default_agent_info_dict = raw_config["agent_dict"][
            custom_agent_info_dict["name"]
        ]
        assignment_agent_info_dict = ConfigUtility._get_custom_instance_info_dict(
            default_agent_info_dict, custom_agent_info_dict
        )
        assignment_agent = GeneralInstanceFactory.model_validate(
            assignment_agent_info_dict
        )
        if (
            language_model_name := assignment_agent.parameters.get("language_model")
        ) is not None:
            # Do not replace the language_model in the parameters with the GeneralInstanceFactory instance.
            assert language_model_name in assignment_language_model_dict
        # endregion
        # region Construct assignment_callback_dict
        assignment_callback_dict: dict[str, Any] = raw_config["assignment_config"][
            "callback_dict"
        ]
        # print("=" * 80)
        # print("DEBUG: Checking callback_dict...")
        # if "callback_dict" in raw_config:
        #     available_callbacks = list(raw_config["callback_dict"].keys())
        #     print(f"Available callbacks ({len(available_callbacks)}): {available_callbacks}")
        # else:
        #     print("ERROR: 'callback_dict' key not found in raw_config!")
        #     print(f"Top-level keys: {list(raw_config.keys())}")
        # print("=" * 80)
        for callback_key, callback_info_dict in assignment_callback_dict.items():
            default_callback_info_dict = raw_config["callback_dict"][
                callback_info_dict["name"]
            ]
            assignment_callback_dict[callback_key] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_callback_info_dict, callback_info_dict
                )
            )
        # endregion
        assignment_config = AssignmentConfig(
            task=raw_config["task_dict"][raw_config["assignment_config"]["task"]],
            agent=assignment_agent,
            language_model_dict=assignment_language_model_dict,
            output_dir=raw_config["assignment_config"]["output_dir"],
            sample_order=raw_config["assignment_config"]["sample_order"],
            callback_dict=assignment_callback_dict,
        )
        # endregion
        # region Convert raw_config into environment_config
        if raw_config["environment_config"]["use_task_client_flag"]:
            environment_config = EnvironmentConfig(
                task_client=raw_config["environment_config"]["task_client"],
                chat_history_item_factory_client=raw_config["environment_config"][
                    "chat_history_item_factory_client"
                ],
                server_side_controller_address=raw_config["environment_config"][
                    "server_side_controller_address"
                ],
                interpreter_path=raw_config["environment_config"]["interpreter_path"],
            )
        else:
            environment_config = EnvironmentConfig(
                task_client=None,
                chat_history_item_factory_client=None,
                server_side_controller_address=None,
                interpreter_path=None,
            )
        # endregion
        # region Convert raw_config into logger_config
        if raw_config["logger_config"]["log_file_path"] == "default":
            if raw_config["environment_config"]["use_task_client_flag"]:
                match caller:
                    case ConfigUtilityCaller.CLIENT:
                        log_file_path = os.path.join(
                            assignment_config.output_dir, "singleton_logger_client.log"
                        )
                    case ConfigUtilityCaller.SERVER:
                        log_file_path = os.path.join(
                            assignment_config.output_dir, "singleton_logger_server.log"
                        )
                    case ConfigUtilityCaller.CLIENT_SIDE_CONTROLLER:
                        log_file_path = (
                            "./outputs/singleton_logger_client_side_controller.log"
                        )
                    case _:
                        raise NotImplementedError()
            else:
                log_file_path = os.path.join(
                    assignment_config.output_dir, "singleton_logger.log"
                )
        else:
            log_file_path = raw_config["logger_config"]["log_file_path"]
        logger_config = LoggerConfig(
            level=raw_config["logger_config"]["level"],
            log_file_path=log_file_path,
            logger_name=raw_config["logger_config"]["logger_name"],
        )
        # endregion
        # region Construct path_config from assignment_config
        path_config = PathConfig(
            exception_record_file_path=os.path.join(
                assignment_config.output_dir, "exception.txt"
            ),
            config_output_path=os.path.join(
                assignment_config.output_dir, "config.yaml"
            ),
            session_list_output_path=os.path.join(
                assignment_config.output_dir, "runs.json"
            ),
            metric_output_path=os.path.join(
                assignment_config.output_dir, "metric.json"
            ),
            coredumpy_output_dir=os.path.join(
                assignment_config.output_dir, "coredumpy"
            ),
        )
        # endregion
        return assignment_config, environment_config, logger_config, path_config

    @staticmethod
    def is_raw_config_equal(
        raw_config_1: dict[str, Any], raw_config_2: dict[str, Any]
    ) -> bool:
        raw_config_1 = copy.deepcopy(raw_config_1)
        raw_config_2 = copy.deepcopy(raw_config_2)
        output_dir_1 = raw_config_1["assignment_config"].pop("output_dir")
        output_dir_2 = raw_config_2["assignment_config"].pop("output_dir")
        return raw_config_1 == raw_config_2 and AssignmentConfig.is_output_dir_equal(
            output_dir_1, output_dir_2
        )


def main() -> None:
    # region Prepare variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples to run.")
    args = parser.parse_args()
    raw_config = ConfigLoader().load_from(args.config_path)
    assignment_config, environment_config, logger_config, path_config = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.CLIENT)
    )
    # Prepare the variable that will be used in the following procedures.
    config_utility = ConfigUtility(assignment_config, environment_config, path_config)
    # endregion
    # region write raw_config to disk
    cleaned_config = config_utility.remove_redundant_args(raw_config)
    config_output_path = path_config.config_output_path
    if os.path.exists(config_output_path):
        config_from_disk = yaml.safe_load(open(config_output_path, "r"))
        assert ConfigUtility.is_raw_config_equal(config_from_disk, cleaned_config)
        # The config file already exists, so we don't need to write it again.
    else:
        # Write the config file to the output directory.
        config_output_dir = os.path.dirname(config_output_path)
        if not os.path.exists(config_output_dir):
            os.makedirs(config_output_dir)
        yaml.dump(
            cleaned_config,
            open(config_output_path, "w"),
        )
    # endregion
    # region Initialize logger, Set coredumpy output dir
    logger = SingletonLogger.get_instance(logger_config)
    coredumpy.patch_except(directory=path_config.coredumpy_output_dir)
    # endregion
    # region Construct variable, valid config
    config_utility.preprocess()
    task, agent, callback_dict = config_utility.construct()
    config_utility.postprocess(task, agent)
    config_utility.validate(task, agent)
    ContinualAgentBenchException.set_record_file(path_config.exception_record_file_path)
    # endregion
    # region Determine whether to start a new assignment or restore the previous incomplete assignment, based on
    # whether the config file exists.
    session_list_output_path = path_config.session_list_output_path
    assert isinstance(assignment_config.sample_order, list)
    session_list: list[Session]
    unfinished_sample_order: list[SampleIndex]
    if os.path.exists(session_list_output_path):
        # At least one session exists, so we restore the previous incomplete assignment.
        session_list = [
            Session.model_validate(session_info_dict)
            for session_info_dict in json.load(open(session_list_output_path, "r"))
        ]
        unfinished_sample_order = [
            sample_index
            for sample_index in assignment_config.sample_order
            if all(session.sample_index != sample_index for session in session_list)
        ]
        # Previous session may change the state of the callback, restore it here.
        CallbackRestorer.restore(callback_dict)
    else:
        # Start a new assignment.
        session_list = []
        unfinished_sample_order = assignment_config.sample_order
    callback_handler = CallbackHandler(callback_dict)
    # 检测是否启用GRPO训练（支持原版、RLLM版本和MemoryReview版本）
    enable_grpo = any(
        (GRPOTrainingCallback is not None and isinstance(cb, GRPOTrainingCallback)) or
        (GRPOTrainingCallbackRLLM is not None and isinstance(cb, GRPOTrainingCallbackRLLM)) or
        (MemoryReviewGRPOCallback is not None and isinstance(cb, MemoryReviewGRPOCallback))
        for cb in callback_dict.values()
    )
    # 检测是否启用 Reranker GRPO 训练
    enable_reranker_grpo = any(
        RerankerGRPOTrainingCallback is not None and isinstance(cb, RerankerGRPOTrainingCallback)
        for cb in callback_dict.values()
    )
    # 检测是否启用 Reflection GRPO 训练
    enable_reflection_grpo = any(
        ReflectionGRPOTrainingCallback is not None and isinstance(cb, ReflectionGRPOTrainingCallback)
        for cb in callback_dict.values()
    )
    # DEBUG: 打印 callback 类型和 enable_grpo 状态
    logger.info(f"[DEBUG] Callbacks loaded: {[(k, type(v).__name__) for k, v in callback_dict.items()]}")
    logger.info(f"[DEBUG] MemoryReviewGRPOCallback class: {MemoryReviewGRPOCallback}")
    logger.info(f"[DEBUG] enable_grpo = {enable_grpo}, enable_reranker_grpo = {enable_reranker_grpo}, "
                f"enable_reflection_grpo = {enable_reflection_grpo}")
    # Apply sample limit if provided
    if args.max_samples is not None and args.max_samples > 0:
        unfinished_sample_order = unfinished_sample_order[: args.max_samples]
    # endregion
    # region Run experiment
    logger.info(
        f"Experiment start. "
        f"Total sample count: {len(assignment_config.sample_order)}. "
        f"Unfinished sample count: {len(unfinished_sample_order)}."
    )
    # region Determine RL sampling config from callbacks (default group_size=1)
    def _get_group_config(callback_dict: Mapping[str, Callback]) -> tuple[int, str]:
        group_size = 1
        strategy = "best_reward"
        for cb in callback_dict.values():
            # 只从普通 GRPO callbacks 获取 group_size，不包括 RerankerGRPOTrainingCallback
            if RerankerGRPOTrainingCallback is not None and isinstance(cb, RerankerGRPOTrainingCallback):
                continue
            if hasattr(cb, "group_size"):
                group_size = max(group_size, getattr(cb, "group_size"))
            if hasattr(cb, "best_metric_strategy"):
                strategy = getattr(cb, "best_metric_strategy")
        return group_size, strategy

    def _get_reranker_group_size(callback_dict: Mapping[str, Callback]) -> int:
        """获取 reranker GRPO 的 group_size"""
        for cb in callback_dict.values():
            if RerankerGRPOTrainingCallback is not None and isinstance(cb, RerankerGRPOTrainingCallback):
                return getattr(cb, "group_size", 4)
        return 4

    def _get_reflection_k1_k2(callback_dict: Mapping[str, Callback]) -> tuple[int, int]:
        """获取 reflection GRPO 的 k1 (reflections per sample) 和 k2 (rollouts per reflection)"""
        for cb in callback_dict.values():
            if ReflectionGRPOTrainingCallback is not None and isinstance(cb, ReflectionGRPOTrainingCallback):
                k1 = getattr(cb, "k1", 4)
                k2 = getattr(cb, "k2", 4)
                return k1, k2
        return 4, 4

    # 奖励函数已移至grpo_training_callback.py中的_calc_reward方法

    group_size, best_metric_strategy = _get_group_config(callback_dict)
    reranker_group_size = _get_reranker_group_size(callback_dict) if enable_reranker_grpo else 1
    reflection_k1, reflection_k2 = _get_reflection_k1_k2(callback_dict) if enable_reflection_grpo else (1, 1)
    if not enable_grpo:
        group_size = 1
    # endregion
    group_correct_record: list[bool] = []
    for sample_index in unfinished_sample_order:
        logger.info(
            f"Sample {sample_index} start with group_size={group_size}, "
            f"reranker_group_size={reranker_group_size}, reflection_k1={reflection_k1}, reflection_k2={reflection_k2} "
            f"(best_metric_strategy={best_metric_strategy})."
        )
        # ----- Greedy evaluation pass (for metric) -----
        original_inference_cfg = getattr(agent, "_inference_config_dict", None)
        max_new_tokens = None
        if isinstance(original_inference_cfg, dict):
            max_new_tokens = original_inference_cfg.get("max_new_tokens")
        greedy_inference_cfg = {
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else 512,
        }
        agent._inference_config_dict = greedy_inference_cfg  # type: ignore[attr-defined]
        setattr(agent, "_force_greedy", True)
        # Temporarily ask GRPO callbacks to skip override
        grpo_callbacks = [
            cb for cb in callback_dict.values()
            if (GRPOTrainingCallback is not None and isinstance(cb, GRPOTrainingCallback)) or
               (GRPOTrainingCallbackRLLM is not None and isinstance(cb, GRPOTrainingCallbackRLLM)) or
               (MemoryReviewGRPOCallback is not None and isinstance(cb, MemoryReviewGRPOCallback))
        ]
        for cb in grpo_callbacks:
            cb.skip_override = True  # type: ignore[attr-defined]

        # ----- BASELINE evaluation pass (for Reranker GRPO: no memory injection) -----
        if enable_reranker_grpo:
            baseline_session = Session(task_name=task.task_name, sample_index=sample_index)
            # 注意：finish_reason 必须在 task.complete() 之前设置，不能在创建时设置
            # 否则 task.interact() 会因为 assert finish_reason is None 而失败
            baseline_callback_args = CallbackArguments(
                current_session=baseline_session,
                task=task,
                agent=agent,
                session_list=session_list,
            )
            # 在 on_session_create 之前标记为 baseline（供 callback 识别）
            setattr(baseline_session, '_is_baseline_eval', True)
            callback_handler.on_session_create(baseline_callback_args)
            if baseline_callback_args.session_controller.should_task_reset:
                task.reset(baseline_session)
                callback_handler.on_task_reset(baseline_callback_args)
            logger.info(f"Sample {sample_index} BASELINE eval start (no memory).")
            while baseline_session.sample_status == SampleStatus.RUNNING:
                if baseline_callback_args.session_controller.should_agent_inference:
                    agent.inference(baseline_session)
                    callback_handler.on_agent_inference(baseline_callback_args)
                if baseline_callback_args.session_controller.should_task_interact:
                    task.interact(baseline_session)
                    callback_handler.on_task_interact(baseline_callback_args)
            if baseline_callback_args.session_controller.should_task_complete:
                baseline_session.finish_reason = "BASELINE_EVAL"  # 在 complete 之前设置
                task.complete(baseline_session)
                callback_handler.on_task_complete(baseline_callback_args)
            logger.info(
                f"Sample {sample_index} BASELINE eval end. "
                f"Session status: {baseline_session.sample_status}. "
                f"Evaluation outcome: {baseline_session.evaluation_record.outcome}."
            )

        # ----- Greedy evaluation pass 1 (for metric, WITH memory) -----
        greedy_session = Session(task_name=task.task_name, sample_index=sample_index)
        greedy_callback_args = CallbackArguments(
            current_session=greedy_session,
            task=task,
            agent=agent,
            session_list=session_list,
        )
        callback_handler.on_session_create(greedy_callback_args)
        if greedy_callback_args.session_controller.should_task_reset:
            task.reset(greedy_session)
            callback_handler.on_task_reset(greedy_callback_args)
        logger.info(f"Sample {sample_index} greedy eval pass 1 start (with memory).")
        while greedy_session.sample_status == SampleStatus.RUNNING:
            if greedy_callback_args.session_controller.should_agent_inference:
                agent.inference(greedy_session)
                callback_handler.on_agent_inference(greedy_callback_args)
            if greedy_callback_args.session_controller.should_task_interact:
                task.interact(greedy_session)
                callback_handler.on_task_interact(greedy_callback_args)
        if greedy_callback_args.session_controller.should_task_complete:
            greedy_session.finish_reason = "GREEDY_EVAL"
            task.complete(greedy_session)
            callback_handler.on_task_complete(greedy_callback_args)

        # 判断是否正确：只要 evaluation outcome 是 CORRECT 就算正确
        # （即使 sample_status 是 task_limit_reached，只要答案对了就算成功）
        greedy_pass1_correct = (
            greedy_session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
        )
        logger.info(
            f"Sample {sample_index} greedy eval pass 1 end. "
            f"Session status: {greedy_session.sample_status}. "
            f"Evaluation outcome: {greedy_session.evaluation_record.outcome}. "
            f"Correct: {greedy_pass1_correct}"
        )

        # ----- Greedy evaluation pass 2 (with reflection, if pass 1 failed) -----
        # 支持两种方式生成 reflection：
        # 1. 使用 reflection_grpo_training_callback（如果存在）
        # 2. 使用 previous_sample_embedding_callback 的 generate_retry_reflection（如果配置了 reflection）
        greedy_pass2_correct = False
        greedy_session_2: Optional[Session] = None  # 用于保存 pass 2 的 session

        # 检查是否有可用的 reflection 生成器
        reflection_grpo_cb = None
        embedding_cb_for_reflection = None
        for cb in callback_dict.values():
            if ReflectionGRPOTrainingCallback is not None and isinstance(cb, ReflectionGRPOTrainingCallback):
                reflection_grpo_cb = cb
            # 检查 previous_sample_embedding_callback 是否配置了 reflection
            from src.callbacks.instance.previous_sample_embedding_callback import PreviousSampleEmbeddingCallback
            if isinstance(cb, PreviousSampleEmbeddingCallback):
                # 检查是否配置了本地模型或 API
                if cb.reflection_use_local_model or os.getenv(cb.reflection_api_key_env, ""):
                    embedding_cb_for_reflection = cb

        # 决定是否启用 retry with reflection
        enable_retry_reflection = (
            not greedy_pass1_correct and
            (enable_reflection_grpo or embedding_cb_for_reflection is not None)
        )

        if enable_retry_reflection:
            # 提取当前 query 和 greedy 轨迹（用于生成 reflection）
            current_query_for_retry = ""
            greedy_trajectory_for_retry = ""
            try:
                candidate = greedy_session.chat_history.get_item_deep_copy(2)
                from src.typings import Role
                if candidate.role == Role.USER:
                    current_query_for_retry = candidate.content.strip()

                agent_role_dict = agent.get_role_dict()
                greedy_trajectory_for_retry = greedy_session.chat_history.get_value_str(
                    agent_role_dict, start_index=3, end_index=None
                )
            except Exception as e:
                logger.warning(f"Failed to extract query/trajectory for retry reflection: {e}")

            retry_insight_text = None
            if current_query_for_retry and greedy_trajectory_for_retry:
                # 优先使用 reflection_grpo_cb（如果存在）
                if reflection_grpo_cb is not None:
                    retry_reflection_text, retry_insight_text, _ = reflection_grpo_cb.generate_reflection(
                        current_query=current_query_for_retry,
                        greedy_trajectory=greedy_trajectory_for_retry,
                        greedy_correct=False,
                        sample_index=sample_index,
                        reflection_id=-1  # 特殊 ID 表示 retry，不用于 GRPO 训练
                    )
                    logger.info(f"[RetryReflection] Using reflection_grpo_cb to generate reflection")
                # 否则使用 embedding_cb_for_reflection
                elif embedding_cb_for_reflection is not None:
                    retry_insight_text = embedding_cb_for_reflection.generate_retry_reflection(
                        query=current_query_for_retry,
                        trajectory=greedy_trajectory_for_retry,
                        outcome="failure",
                        error_message="Wrong answer",
                    )
                    logger.info(f"[RetryReflection] Using previous_sample_embedding_callback to generate reflection")

                if retry_insight_text:
                    logger.info(
                        f"Sample {sample_index} generated retry reflection: {retry_insight_text[:100]}..."
                    )

                    # ----- Greedy evaluation pass 2 -----
                    greedy_session_2 = Session(task_name=task.task_name, sample_index=sample_index)
                    # 设置 reflection 到 session，用于注入到第二个 user prompt（task query）
                    setattr(greedy_session_2, '_retry_reflection_text', retry_insight_text)
                    setattr(greedy_session_2, '_is_retry_with_reflection', True)

                    greedy_callback_args_2 = CallbackArguments(
                        current_session=greedy_session_2,
                        task=task,
                        agent=agent,
                        session_list=session_list,
                    )
                    callback_handler.on_session_create(greedy_callback_args_2)
                    if greedy_callback_args_2.session_controller.should_task_reset:
                        task.reset(greedy_session_2)
                        callback_handler.on_task_reset(greedy_callback_args_2)

                    logger.info(f"Sample {sample_index} greedy eval pass 2 start (with reflection).")
                    while greedy_session_2.sample_status == SampleStatus.RUNNING:
                        if greedy_callback_args_2.session_controller.should_agent_inference:
                            agent.inference(greedy_session_2)
                            callback_handler.on_agent_inference(greedy_callback_args_2)
                        if greedy_callback_args_2.session_controller.should_task_interact:
                            task.interact(greedy_session_2)
                            callback_handler.on_task_interact(greedy_callback_args_2)
                    if greedy_callback_args_2.session_controller.should_task_complete:
                        greedy_session_2.finish_reason = "GREEDY_EVAL_RETRY"
                        task.complete(greedy_session_2)
                        callback_handler.on_task_complete(greedy_callback_args_2)

                    # 判断是否正确：只要 evaluation outcome 是 CORRECT 就算正确
                    # （即使 sample_status 是 task_limit_reached，只要答案对了就算成功）
                    greedy_pass2_correct = (
                        greedy_session_2.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
                    )
                    logger.info(
                        f"Sample {sample_index} greedy eval pass 2 end. "
                        f"Session status: {greedy_session_2.sample_status}. "
                        f"Evaluation outcome: {greedy_session_2.evaluation_record.outcome}. "
                        f"Correct: {greedy_pass2_correct}"
                    )
                    # greedy_session_2 已在上面被赋值，无需再更新

        # Record metric: pass 1 或 pass 2 任一成功都算正确
        sample_any_correct = greedy_pass1_correct or greedy_pass2_correct
        logger.info(
            f"Sample {sample_index} final metric: pass1={greedy_pass1_correct}, pass2={greedy_pass2_correct}, "
            f"any_correct={sample_any_correct}"
        )

        # Note: Reflection for greedy trajectory will be automatically generated by
        # previous_sample_embedding_callback.on_state_save() if enable_reflection=True
        # 保存 pass 1 的 session
        session_list.append(greedy_session)
        # 如果执行了 pass 2，也保存 pass 2 的 session
        if greedy_session_2 is not None:
            session_list.append(greedy_session_2)
        group_correct_record.append(sample_any_correct)
        # Persist current session list
        json.dump(
            [s.model_dump() for s in session_list],
            open(session_list_output_path, "w"),  # noqa
            indent=2,
        )
        # Restore inference config for sampling
        agent._inference_config_dict = original_inference_cfg  # type: ignore[attr-defined]
        if hasattr(agent, "_force_greedy"):
            delattr(agent, "_force_greedy")
        for cb in grpo_callbacks:
            cb.skip_override = False  # type: ignore[attr-defined]

        # ----- Reflection GRPO: nested k1 reflections × k2 base rollouts -----
        # 这里训练 reflection model（外层）和 base model（内层，复用 grpo_callbacks）
        if enable_reflection_grpo:
            # 注册 greedy 结果到 reflection callback
            # 使用 pass1 的结果（因为 reflection 是基于 pass1 的轨迹生成的）
            greedy_correct = greedy_pass1_correct
            reflection_grpo_cb = None
            for cb in callback_dict.values():
                if ReflectionGRPOTrainingCallback is not None and isinstance(cb, ReflectionGRPOTrainingCallback):
                    cb.register_greedy_result(sample_index, greedy_correct)
                    reflection_grpo_cb = cb
                    break

            # 提取当前 query 和 greedy 轨迹（用于生成 reflection）
            current_query = ""
            greedy_trajectory = ""
            try:
                candidate = greedy_session.chat_history.get_item_deep_copy(2)
                from src.typings import Role
                if candidate.role == Role.USER:
                    current_query = candidate.content.strip()

                # 提取 greedy 轨迹（从 index 3 开始，包含 agent 的完整推理过程）
                agent_role_dict = agent.get_role_dict()
                greedy_trajectory = greedy_session.chat_history.get_value_str(
                    agent_role_dict, start_index=3, end_index=None
                )
            except Exception as e:
                logger.warning(f"Failed to extract query/trajectory for reflection: {e}")

            # 外层循环：k1 个 reflections
            for reflection_id in range(reflection_k1):
                # 生成 reflection（基于 greedy 轨迹）
                reflection_text = ""  # 完整的 JSON（用于存储和训练）
                insight_text = ""     # 提取的 insight（用于注入到 prompt）
                reflection_messages = []
                if reflection_grpo_cb is not None and current_query and greedy_trajectory:
                    reflection_text, insight_text, reflection_messages = reflection_grpo_cb.generate_reflection(
                        current_query=current_query,
                        greedy_trajectory=greedy_trajectory,
                        greedy_correct=greedy_correct,
                        sample_index=sample_index,
                        reflection_id=reflection_id
                    )
                    # 注册 reflection 生成（用于训练）
                    if reflection_text and reflection_messages:
                        reflection_grpo_cb.register_reflection_generation(
                            sample_index, reflection_id, reflection_messages, reflection_text
                        )
                    logger.info(
                        f"Sample {sample_index} reflection {reflection_id + 1}/{reflection_k1} generated"
                    )
                    logger.info(f"  Insight: {insight_text[:100]}...")

                # 内层循环：每个 reflection 对应 k2 个 base model rollouts
                for rollout_id in range(reflection_k2):
                    # 为了让 grpo_training_callback_rllm 能够正确分组，
                    # 我们需要让每个 reflection 的 rollouts 看起来像不同的 sample
                    # 使用 (sample_index, reflection_id) 作为虚拟 sample_index
                    virtual_sample_index = f"{sample_index}_r{reflection_id}"
                    session = Session(task_name=task.task_name, sample_index=virtual_sample_index)
                    # 同时保存原始 sample_index，供其他 callback 使用
                    setattr(session, '_original_sample_index', sample_index)
                    # 保存 insight_text 到 session，供 callback 注入到 prompt
                    # 同时保存完整的 reflection_text 和 messages（用于训练）
                    setattr(session, '_reflection_text', insight_text)  # 注入用
                    setattr(session, '_reflection_full', reflection_text)  # 完整 JSON（可选）
                    setattr(session, '_reflection_messages', reflection_messages)
                    callback_args = CallbackArguments(
                        current_session=session,
                        task=task,
                        agent=agent,
                        session_list=session_list,
                    )
                    # 设置 reflection_id 到 session，供 callback 使用
                    setattr(session, '_reflection_id', reflection_id)
                    setattr(session, '_rollout_id', rollout_id)

                    callback_handler.on_session_create(callback_args)
                    if callback_args.session_controller.should_task_reset:
                        # 临时恢复原始 sample_index 用于 task.reset (dataset lookup)
                        # 保存 virtual index
                        virtual_index = session.sample_index
                        # 使用原始 index 进行 reset
                        session.sample_index = sample_index
                        task.reset(session)
                        # 立即恢复 virtual index (在 on_task_reset 之前)
                        session.sample_index = virtual_index
                        # 同时更新 task.current_sample_index 为 virtual index (避免 interact/complete 时的 assert 失败)
                        task.current_sample_index = virtual_index
                        callback_handler.on_task_reset(callback_args)
                    logger.info(
                        f"Sample {sample_index} reflection {reflection_id + 1}/{reflection_k1} "
                        f"rollout {rollout_id + 1}/{reflection_k2} start."
                    )
                    while session.sample_status == SampleStatus.RUNNING:
                        if callback_args.session_controller.should_agent_inference:
                            agent.inference(session)
                            callback_handler.on_agent_inference(callback_args)
                        if callback_args.session_controller.should_task_interact:
                            task.interact(session)
                            callback_handler.on_task_interact(callback_args)
                    if callback_args.session_controller.should_task_complete:
                        task.complete(session)
                        callback_handler.on_task_complete(callback_args)

                    # 注册 rollout 结果到 reflection callback
                    # 只看 evaluation outcome，不需要检查 sample_status
                    # 因为 task_limit_reached 等状态下也可能是正确的结果
                    rollout_correct = session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
                    for cb in callback_dict.values():
                        if ReflectionGRPOTrainingCallback is not None and isinstance(cb, ReflectionGRPOTrainingCallback):
                            cb.register_rollout_result(sample_index, reflection_id, rollout_correct, session)
                            break

                    logger.info(
                        f"Sample {sample_index} reflection {reflection_id + 1}/{reflection_k1} "
                        f"rollout {rollout_id + 1}/{reflection_k2} end. "
                        f"Session status: {session.sample_status}. "
                        f"Evaluation outcome: {session.evaluation_record.outcome}."
                    )

            # 检查是否需要存储成功轨迹到 memory（如果 greedy 失败但有 rollout 成功）
            for cb in callback_dict.values():
                if ReflectionGRPOTrainingCallback is not None and isinstance(cb, ReflectionGRPOTrainingCallback):
                    # 检查是否启用了成功轨迹存储
                    if cb.store_rollout_success_to_memory:
                        success_candidate = cb.get_success_candidate_for_memory(sample_index)
                        if success_candidate is not None:
                            # 存储到 memory（通过 previous_sample_embedding_callback）
                            try:
                                from src.callbacks.instance.previous_sample_embedding_callback import PreviousSampleEmbeddingCallback
                                for mem_cb in callback_dict.values():
                                    if isinstance(mem_cb, PreviousSampleEmbeddingCallback):
                                        # 将成功轨迹加入待存储队列（会在 on_state_save 时真正存储）
                                        mem_cb._pending_session_to_store = (success_candidate, "success", "")
                                        logger.info(
                                            f"[ReflectionGRPO] Queued success trajectory from rollout for sample {sample_index} "
                                            f"(greedy failed, rollout succeeded)"
                                        )
                                        break
                            except ImportError:
                                pass
                            # 清理候选
                            cb.clear_success_candidate(sample_index)
                    break

        # ----- GRPO sampling/training passes (not used for metric) -----
        # 注意：如果启用了 Reflection GRPO，则跳过独立的 GRPO rollouts
        # 因为 Reflection GRPO 的 k2 rollouts 已经会触发 grpo_training_callback_rllm 的训练
        last_training_session: Optional[Session] = None
        if enable_grpo and not enable_reflection_grpo:
            for attempt_id in range(group_size):
                session = Session(task_name=task.task_name, sample_index=sample_index)
                callback_args = CallbackArguments(
                    current_session=session,
                    task=task,
                    agent=agent,
                    session_list=session_list,
                )
                callback_handler.on_session_create(callback_args)
                if callback_args.session_controller.should_task_reset:
                    task.reset(session)
                    callback_handler.on_task_reset(callback_args)
                logger.info(f"Sample {sample_index} training attempt {attempt_id + 1}/{group_size} start.")
                while session.sample_status == SampleStatus.RUNNING:
                    if callback_args.session_controller.should_agent_inference:
                        agent.inference(session)
                        callback_handler.on_agent_inference(callback_args)
                    if callback_args.session_controller.should_task_interact:
                        task.interact(session)
                        callback_handler.on_task_interact(callback_args)
                if callback_args.session_controller.should_task_complete:
                    task.complete(session)
                    callback_handler.on_task_complete(callback_args)
                logger.info(
                    f"Sample {sample_index} training attempt {attempt_id + 1}/{group_size} end. "
                    f"Session status: {session.sample_status}. "
                    f"Evaluation outcome: {session.evaluation_record.outcome}."
                )
                last_training_session = session

        # ----- Reranker GRPO sampling/training passes -----
        # 每次采样时 reranker 可能选择不同的记忆组合，用于训练 reranker 模型
        # 获取 warmup 状态，warmup 期间跳过 rollout
        reranker_in_warmup = False
        if enable_reranker_grpo:
            for cb in callback_dict.values():
                if RerankerGRPOTrainingCallback is not None and isinstance(cb, RerankerGRPOTrainingCallback):
                    reranker_in_warmup = cb._processed_samples < cb.warmup_samples
                    break

        if enable_reranker_grpo and not reranker_in_warmup:
            for attempt_id in range(reranker_group_size):
                session = Session(task_name=task.task_name, sample_index=sample_index)
                callback_args = CallbackArguments(
                    current_session=session,
                    task=task,
                    agent=agent,
                    session_list=session_list,
                )
                callback_handler.on_session_create(callback_args)
                if callback_args.session_controller.should_task_reset:
                    task.reset(session)
                    callback_handler.on_task_reset(callback_args)
                logger.info(f"Sample {sample_index} reranker training attempt {attempt_id + 1}/{reranker_group_size} start.")
                while session.sample_status == SampleStatus.RUNNING:
                    if callback_args.session_controller.should_agent_inference:
                        agent.inference(session)
                        callback_handler.on_agent_inference(callback_args)
                    if callback_args.session_controller.should_task_interact:
                        task.interact(session)
                        callback_handler.on_task_interact(callback_args)
                if callback_args.session_controller.should_task_complete:
                    task.complete(session)
                    callback_handler.on_task_complete(callback_args)
                logger.info(
                    f"Sample {sample_index} reranker training attempt {attempt_id + 1}/{reranker_group_size} end. "
                    f"Session status: {session.sample_status}. "
                    f"Evaluation outcome: {session.evaluation_record.outcome}."
                )
                last_training_session = session
        elif enable_reranker_grpo and reranker_in_warmup:
            # Warmup 期间：只更新计数器，不做 rollout
            for cb in callback_dict.values():
                if RerankerGRPOTrainingCallback is not None and isinstance(cb, RerankerGRPOTrainingCallback):
                    if sample_index not in cb._seen_sample_indices:
                        cb._seen_sample_indices.add(sample_index)
                        cb._processed_samples += 1
                    logger.info(f"[RerankerGRPO] Warmup: {cb._processed_samples}/{cb.warmup_samples} samples, skipping rollout")
                    break
        # Save callback state based on last training attempt (or greedy if no training run)
        state_session = last_training_session if last_training_session is not None else greedy_session
        callback_args = CallbackArguments(
            current_session=state_session,
            task=task,
            agent=agent,
            session_list=session_list,
        )
        callback_handler.on_state_save(callback_args)
    # endregion
    # region Evaluate
    session_metric_calculation_partial_list: Sequence[
        SessionMetricCalculationPartial
    ] = [
        SessionMetricCalculationPartial(
            sample_index=session.sample_index,
            evaluation_record=session.evaluation_record,
            sample_status=session.sample_status,
        )
        for session in session_list
    ]
    metric = task.calculate_metric(session_metric_calculation_partial_list)
    # Custom grouped accuracy: a sample is correct if any attempt was correct
    total_samples = len(group_correct_record)
    correct_samples = sum(1 for flag in group_correct_record if flag)
    group_accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
    metric["group_sample_accuracy"] = {
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "accuracy": group_accuracy,
    }
    logger.info(
        f"Experiment end. Metric: {metric}. Total sample count: {len(assignment_config.sample_order)}.",
    )
    json.dump(
        metric,
        open(path_config.metric_output_path, "w"),  # noqa
        indent=2,
    )
    logger.info(f"Metric file has been saved to {assignment_config.output_dir}.")
    # endregion
    # region Release
    task.release()
    # endregion


if __name__ == "__main__":
    main()
