import logging
import time
import traceback
from typing import Iterator, List, Optional

from mobile_use.agents import Agent
from mobile_use.agents.color_mobile import (
    ColorMobileAppRetriever,
    ColorMobileMemory,
    ColorMobileOperator,
    ColorMobilePlanner,
)
from mobile_use.agents.color_mobile.app_mapping import format_app_names_for_planner
from mobile_use.schema.config import AppRetrieverConfig, ColorMobileAgentConfig, OperatorConfig, PlannerConfig
from mobile_use.schema.schema import (
    Action,
    AgentState,
    AgentStatus,
    MobileUseEpisodeData,
    MobileUseStepData,
    SingleAgentStepData,
)
from mobile_use.utils.vlm import VLMWrapper

logger = logging.getLogger(__name__)


@Agent.register("ColorMobileAgent")
class ColorMobileAgent(Agent):
    """A compact Planner + Operator + Memory mobile agent."""

    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)
        if config_path is not None:
            self.config = ColorMobileAgentConfig.from_yaml(config_path)
        else:
            self.config = ColorMobileAgentConfig(**kwargs)

        self.app_retriever = self._init_app_retriever(self.config.app_retriever)
        self.planner = self._init_planner(self.config.planner)
        self.operator = self._init_operator(self.config.operator)
        self.memory = ColorMobileMemory(
            max_history_steps=self.config.memory_max_history_steps,
            compress_every_steps=self.config.memory_compress_every_steps,
            recent_steps_after_compress=self.config.memory_recent_steps_after_compress,
            vlm=VLMWrapper(**self.config.vlm.model_dump()) if self.config.vlm else None,
            prompt_config=self.config.memory_prompt_config,
        )
        self.max_action_retry = self.config.max_action_retry

    def _init_data(self, goal: str = ""):
        super()._init_data(goal)
        self.trajectory: List[MobileUseStepData] = []
        self.episode_data: MobileUseEpisodeData = MobileUseEpisodeData(
            goal=goal,
            num_steps=0,
            trajectory=self.trajectory,
        )

    def _init_planner(self, config: Optional[PlannerConfig]) -> Optional[ColorMobilePlanner]:
        if config and config.enabled:
            if config.vlm is None:
                config.vlm = self.config.vlm
            return ColorMobilePlanner(config)
        return None

    def _init_app_retriever(self, config: Optional[AppRetrieverConfig]) -> Optional[ColorMobileAppRetriever]:
        if config and config.enabled:
            if config.vlm is None:
                config.vlm = self.config.vlm
            return ColorMobileAppRetriever(
                vlm=VLMWrapper(**config.vlm.model_dump()),
                prompt_config=config.prompt_config or "color_mobile_app_retriever.yaml",
                enabled=config.enabled,
                chunk_size=config.chunk_size,
                max_apps=config.max_apps,
                max_workers=config.max_workers,
            )
        return None

    def _init_operator(self, config: OperatorConfig) -> ColorMobileOperator:
        if config.vlm is None:
            config.vlm = self.config.vlm
        config.enabled = True
        return ColorMobileOperator(config)

    def reset(self, goal: str = "", max_steps: int = None) -> None:
        self._init_data(goal=goal)
        self.status = None
        self.state = AgentState.RUNNING
        self.memory.reset()
        if isinstance(max_steps, int):
            self.set_max_steps(max_steps)
        if self.planner:
            self.planner.reset()
        self.operator.reset()

    def _get_curr_step_data(self) -> Optional[MobileUseStepData]:
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        return None

    def _plan(self) -> None:
        if not self.planner:
            return

        step_data = self.trajectory[-1]
        if self.app_retriever:
            self.episode_data.installed_apps = self.app_retriever.format_for_planner(self.episode_data.goal)
        else:
            self.episode_data.installed_apps = format_app_names_for_planner()
        response = self.planner.vlm.predict(self.planner.get_message(self.episode_data))
        raw_plan = response.choices[0].message.content
        logger.info("Plan from VLM:\n%s", raw_plan)
        _, plan, current_subgoal = self.planner.parse_response(raw_plan)
        step_data.plan = plan
        step_data.sub_goal = current_subgoal

    def _operate(self) -> None:
        step_data = self.trajectory[-1]
        self.episode_data.recent_history = self.memory.get_recent_history(self.trajectory[:-1])
        self.episode_data.long_term_history = self.memory.get_long_term_history()
        operator_messages = self.operator.get_message(self.episode_data)
        response = self.operator.vlm.predict(operator_messages)

        last_error = None
        for _ in range(self.max_action_retry):
            try:
                raw_action = response.choices[0].message.content
                logger.info("Action from VLM:\n%s", raw_action)
                thought, action, action_s, action_desc = self.operator.parse_response(raw_action)
                step_data.thought = thought
                step_data.action = action
                step_data.action_s = action_s
                step_data.action_desc = action_desc
                return
            except Exception as exc:
                last_error = exc
                logger.warning("Failed to parse action: %s", exc)
                operator_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"动作格式解析失败：{exc}。请严格返回 thought: ...###reasoning: ...###action: ... 格式。",
                            }
                        ],
                    }
                )
                response = self.operator.vlm.predict(operator_messages)

        raise ValueError("Action parse error after max retry.") from last_error

    def step(self) -> MobileUseStepData:
        start_time = time.time()
        logger.info("ColorMobileAgent step %d", self.curr_step_idx)

        env_state = self.env.get_state()
        self.trajectory.append(
            MobileUseStepData(
                step_idx=self.curr_step_idx,
                curr_env_state=env_state,
            )
        )
        step_data = self.trajectory[-1]

        if self.planner and self.curr_step_idx == 0:
            self._plan()
        elif len(self.trajectory) > 1:
            previous_step = self.trajectory[-2]
            step_data.plan = previous_step.plan
            step_data.sub_goal = previous_step.sub_goal

        self._operate()

        action = step_data.action
        if action is None:
            self.status = AgentStatus.FAILED
        elif action.name == "terminate":
            status = action.parameters.get("status") if action.parameters else None
            self.status = AgentStatus.FINISHED if status == "success" else AgentStatus.FAILED
        elif action.name == "answer":
            step_data.answer = action.parameters.get("text") if action.parameters else ""
            self.status = AgentStatus.FINISHED
        elif action.name == "call_user":
            step_data.answer = action.parameters.get("text") if action.parameters else ""
            self.state = AgentState.CALLUSER
        else:
            try:
                start_exec_time = time.time()
                if action.name == "type" and action.parameters and action.parameters.get("coordinate"):
                    self.env.execute_action(Action(name="click", parameters={"coordinate": action.parameters["coordinate"]}))
                self.env.execute_action(action)
                step_data.exec_duration = time.time() - start_exec_time
            except Exception:
                logger.warning("Failed to execute action %s: %s", action, traceback.format_exc())
                self.status = AgentStatus.FAILED

        step_data.exec_env_state = self.env.get_state()
        self.episode_data.memory = self.memory.update(step_data)
        self.memory.maybe_compress(self.trajectory, goal=self.goal, plan=step_data.plan or "")
        self.episode_data.long_term_history = self.memory.get_long_term_history()
        self.episode_data.recent_history = self.memory.get_recent_history(self.trajectory)
        step_data.memory = self.episode_data.memory
        step_data.step_duration = time.time() - start_time
        return step_data

    def iter_run(self, input_content: str) -> Iterator[MobileUseStepData]:
        if self.state == AgentState.READY:
            self.reset(goal=input_content)
            logger.info("Start task: %s, with at most %d steps", self.goal, self.max_steps)
        elif self.state == AgentState.CALLUSER:
            self.state = AgentState.RUNNING
            logger.info("Continue task: %s, with user input %s", self.goal, input_content)
        else:
            raise Exception("Error agent state")

        for step_idx in range(self.curr_step_idx, self.max_steps):
            self.curr_step_idx = step_idx
            yield SingleAgentStepData(
                step_idx=self.curr_step_idx,
                curr_env_state=self.env.get_state(),
                vlm_call_history=[],
            )

            try:
                self.step()
            except Exception as exc:
                self.status = AgentStatus.FAILED
                self.episode_data.status = self.status
                self.episode_data.message = str(exc)
                yield self._get_curr_step_data()
                return

            self.episode_data.num_steps = step_idx + 1
            self.episode_data.status = self.status

            yield self._get_curr_step_data()
            if self.state == AgentState.CALLUSER:
                self.episode_data.message = "Agent asks user for help"
                return
            if self.status in [AgentStatus.FINISHED, AgentStatus.FAILED]:
                self.episode_data.message = "Agent stopped with status %s" % self.status.value
                return

        logger.warning("ColorMobileAgent reached max number of steps: %s.", self.max_steps)

    def run(self, input_content: str) -> MobileUseEpisodeData:
        for _ in self.iter_run(input_content):
            pass
        return self.episode_data
