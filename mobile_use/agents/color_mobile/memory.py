from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import List

import yaml

from mobile_use.agents.color_mobile.parser import ColorMobileActionParser
from mobile_use.schema.schema import MobileUseStepData
from mobile_use.utils.vlm import VLMWrapper


@dataclass
class ColorMobileMemory:
    """In-process memory with periodic compressed long-term history."""

    max_history_steps: int = 10
    compress_every_steps: int = 2
    recent_steps_after_compress: int = 2
    vlm: VLMWrapper = None
    prompt_config: str = "color_mobile_memory.yaml"
    notes: List[str] = field(default_factory=list)
    long_term_history: str = ""
    _last_compressed_step: int = 0
    _compressed_start_step: int = 0

    def __post_init__(self):
        self.prompt = self._load_prompt(self.prompt_config)

    @staticmethod
    def _load_prompt(prompt_config: str) -> dict:
        prompt_path = Path(__file__).parents[2] / "default_prompts" / prompt_config
        with open(prompt_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def reset(self) -> None:
        self.notes.clear()
        self.long_term_history = ""
        self._last_compressed_step = 0
        self._compressed_start_step = 0

    @staticmethod
    def _remove_coords_from_action_string(content: str) -> str:
        patterns = (
            (r"\b(CLICK|click)\[\d+,\d+\]", "CLICK"),
            (r"\b(SWIPE|swipe)\[\d+,\d+,\d+,\d+\]", "SWIPE"),
            (r"\b(LONG_PRESS|long_press)\[\d+,\d+\]", "LONG_PRESS"),
            (r"\b(DOUBLE_CLICK|double_click)\[\d+,\d+\]", "DOUBLE_CLICK"),
            (r"\b(TYPE|type)\[\d+,\d+,(.*?)\]", r"TYPE[\2]"),
        )
        result = content
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE | re.DOTALL)
        return result

    @staticmethod
    def _dedupe_remember(content: str) -> str:
        seen = set()

        def replace(match):
            remember = match.group(1).strip()
            if remember in seen:
                return ""
            seen.add(remember)
            return f"###REMEMBER: {remember}"

        return re.sub(r"###REMEMBER:\s*([^#]+?)(?=###|$)", replace, content, flags=re.IGNORECASE | re.DOTALL)

    @staticmethod
    def _reorder_thought_reasoning(content: str) -> str:
        thought_match = re.search(r"###thought:\s*(.*?)(?=###reasoning:|###action:|###REMEMBER:|$)", content, re.IGNORECASE | re.DOTALL)
        reasoning_match = re.search(r"###reasoning:\s*(.*?)(?=###thought:|###action:|###REMEMBER:|$)", content, re.IGNORECASE | re.DOTALL)
        if not thought_match or not reasoning_match or thought_match.start() > reasoning_match.start():
            return content

        thought = thought_match.group(1).strip()
        reasoning = reasoning_match.group(1).strip()
        content = re.sub(r"###thought:\s*.*?(?=###reasoning:|###action:|###REMEMBER:|$)", "", content, count=1, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r"###reasoning:\s*.*?(?=###thought:|###action:|###REMEMBER:|$)", "", content, count=1, flags=re.IGNORECASE | re.DOTALL)
        return f"{content}###reasoning: {reasoning}###thought: {thought}"

    @staticmethod
    def _strip_next_action_hint(content: str) -> str:
        return re.sub(r"###下一步可能动作[：:][^#]*(?=###|$)", "", content, flags=re.IGNORECASE | re.DOTALL).strip()

    @staticmethod
    def _format_step(
        step_data: MobileUseStepData,
        include_thought: bool = False,
        include_coords: bool = False,
        include_remember: bool = False,
    ) -> str:
        action_text = step_data.action_s or (str(step_data.action) if step_data.action else "无动作")
        action_text = ColorMobileMemory._strip_next_action_hint(action_text)
        action_text = ColorMobileMemory._reorder_thought_reasoning(action_text)
        if not include_coords:
            action_text = ColorMobileMemory._remove_coords_from_action_string(action_text)
        if not include_remember:
            action_text = re.sub(r"###REMEMBER:\s*[^#]+?(?=###|$)", "", action_text, flags=re.IGNORECASE | re.DOTALL)
        action_text = ColorMobileMemory._dedupe_remember(action_text)

        parts = [f"Step{step_data.step_idx + 1}：{action_text}"]
        if step_data.action_desc:
            parts.append(f"###reasoning: {step_data.action_desc}")
        if include_thought and step_data.thought:
            parts.append(f"###thought: {step_data.thought}")
        remember = ColorMobileActionParser.extract_remember(step_data.action_s or "")
        if include_remember and remember and "###REMEMBER:" not in action_text:
            parts.append(f"###REMEMBER: {remember}")
        return ColorMobileMemory._dedupe_remember("".join(parts))

    @staticmethod
    def _parse_summary_response(response_content: str, start_step: int, end_step: int) -> str:
        content = response_content.strip()
        summary_content = content
        action_trajectory = ""
        forbidden_paths = []

        if "[行动轨迹]" in content:
            before, after = content.split("[行动轨迹]", 1)
            summary_content = before.replace("[总结内容]", "").strip()
            if "[绝不可尝试的道路]" in after:
                trajectory_text, forbidden_text = after.split("[绝不可尝试的道路]", 1)
            else:
                trajectory_text, forbidden_text = after, ""
            action_trajectory = re.sub(r"\s+", " ", trajectory_text).strip()
            for line in forbidden_text.splitlines():
                item = line.strip().lstrip("- ").strip()
                if item and item != "无" and "无" not in item[:4]:
                    forbidden_paths.append(item)
        elif "[绝不可尝试的道路]" in content:
            summary_content, forbidden_text = content.split("[绝不可尝试的道路]", 1)
            summary_content = summary_content.replace("[总结内容]", "").strip()
            for line in forbidden_text.splitlines():
                item = line.strip().lstrip("- ").strip()
                if item and item != "无" and "无" not in item[:4]:
                    forbidden_paths.append(item)
        else:
            summary_content = content.replace("[总结内容]", "").strip()

        formatted = f"对step{start_step}~step{end_step}步的总结：\n{summary_content or '无'}"
        if action_trajectory:
            formatted += f"\n行动轨迹：\n{action_trajectory}"
        if forbidden_paths:
            formatted += "\n绝不可尝试的道路：\n" + "\n".join(f"  - {path}" for path in forbidden_paths)
        return formatted

    def update(self, step_data: MobileUseStepData) -> str:
        remember = ColorMobileActionParser.extract_remember(step_data.action_s or "")
        if remember:
            self.notes.append(remember)
        elif step_data.action and step_data.action.name == "take_note":
            text = step_data.action.parameters.get("text") if step_data.action.parameters else None
            if text:
                self.notes.append(text.strip())

        if step_data.action_desc:
            self.notes.append(f"Step {step_data.step_idx + 1}: {step_data.action_desc}")

        self.notes = self.notes[-self.max_history_steps :]
        return self.get_recent_notes()

    def get_recent_notes(self) -> str:
        return "\n".join(self.notes)

    @staticmethod
    def _strip_summary_header(content: str) -> str:
        return re.sub(r"^对step\d+~step\d+步的总结：\s*", "", content.strip(), count=1)

    @classmethod
    def _merge_fallback_summary(cls, previous_summary: str, new_summary: str, start_step: int, end_step: int) -> str:
        previous_body = cls._strip_summary_header(previous_summary)
        new_body = cls._strip_summary_header(new_summary)
        body = "\n".join(part for part in (previous_body, new_body) if part).strip()
        return f"对step{start_step}~step{end_step}步的总结：\n{body or '无'}"

    def _compress_with_model(
        self,
        goal: str,
        recent_history: str,
        start_step: int,
        end_step: int,
        summary_start_step: int,
        summary_end_step: int,
        plan: str = "",
    ) -> str:
        if self.vlm is None:
            raise RuntimeError("Memory VLM is not configured.")

        user_prompt = self.prompt["user_prompt"].format(
            goal=goal,
            plan=plan or "无",
            current_memory=self.long_term_history or "无",
            recent_history=recent_history or "无",
            recent_notes=self.get_recent_notes() or "无",
            start_step=start_step,
            end_step=end_step,
            summary_start_step=summary_start_step,
            summary_end_step=summary_end_step,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.prompt["system_prompt"]}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        response = self.vlm.predict(messages)
        return self._parse_summary_response(response.choices[0].message.content, summary_start_step, summary_end_step)

    def maybe_compress(self, trajectory: List[MobileUseStepData], goal: str = "", plan: str = "") -> str:
        if self.compress_every_steps <= 0:
            return self.long_term_history

        completed_steps = len(trajectory)
        compress_until = max(0, completed_steps - self.recent_steps_after_compress)
        compressible_steps = compress_until - self._last_compressed_step
        if compressible_steps < self.compress_every_steps:
            return self.long_term_history

        if compress_until <= self._last_compressed_step:
            return self.long_term_history

        steps_to_compress = trajectory[self._last_compressed_step:compress_until]
        if not steps_to_compress:
            return self.long_term_history

        start_step = steps_to_compress[0].step_idx + 1
        end_step = steps_to_compress[-1].step_idx + 1
        summary_start_step = self._compressed_start_step or start_step
        summary_end_step = end_step
        recent_history = "\n".join(
            self._format_step(step, include_thought=True, include_coords=True, include_remember=True)
            for step in steps_to_compress
        )
        try:
            self.long_term_history = self._compress_with_model(
                goal=goal,
                recent_history=recent_history,
                start_step=start_step,
                end_step=end_step,
                summary_start_step=summary_start_step,
                summary_end_step=summary_end_step,
                plan=plan,
            )
        except Exception:
            fallback = self._parse_summary_response(recent_history, start_step, end_step)
            self.long_term_history = (
                self._merge_fallback_summary(self.long_term_history, fallback, summary_start_step, summary_end_step)
                if self.long_term_history
                else fallback
            )
        self._last_compressed_step = compress_until
        self._compressed_start_step = summary_start_step
        return self.long_term_history

    def get_recent_history(self, trajectory: List[MobileUseStepData]) -> str:
        start_idx = max(self._last_compressed_step, len(trajectory) - self.max_history_steps)
        recent_steps = trajectory[start_idx:]
        if not recent_steps:
            return "无"
        return "\n".join(
            self._format_step(step, include_thought=True, include_coords=False, include_remember=False)
            for step in recent_steps
        )

    def get_long_term_history(self) -> str:
        return self.long_term_history or "无"
