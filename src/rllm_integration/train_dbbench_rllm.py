import hydra

from rllm.trainer.agent_trainer import AgentTrainer

from src.rllm_integration.dbbench_agent import DBBenchAgent, load_prelude_messages
from src.rllm_integration.dbbench_dataset import load_dbbench_standard
from src.rllm_integration.dbbench_env import DBBenchEnv


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(cfg):
    train_dataset = load_dbbench_standard()
    prelude_messages = load_prelude_messages()

    cfg.rllm.agent.name = "dbbench_agent"
    cfg.rllm.agent.max_steps = 3
    cfg.rllm.agent.agent_args = {"prelude_messages": prelude_messages}
    cfg.rllm.env.name = "dbbench_env"
    cfg.rllm.env.env_args = {"max_turns": 3}
    cfg.rllm.disable_thinking = True

    trainer = AgentTrainer(
        agent_class=DBBenchAgent,
        env_class=DBBenchEnv,
        agent_args={"prelude_messages": prelude_messages},
        env_args={"max_turns": 3},
        config=cfg,
        train_dataset=train_dataset,
        val_dataset=train_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
