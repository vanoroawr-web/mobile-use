from typing import List

from mobile_use.agents.color_mobile.parser import ColorMobileActionParser
from mobile_use.agents.sub_agent import Operator
from mobile_use.schema.schema import MobileUseEpisodeData, MobileUseStepData
from mobile_use.utils.utils import encode_image_url, generate_message, resize_image


class ColorMobileOperator(Operator):
    """ ColorMobileAgent operator."""

    COORDINATE_SIZE = (999, 999)

    def __init__(self, config):
        super().__init__(config)
        self.parser = ColorMobileActionParser()

    @staticmethod
    def _format_history(trajectory: List[MobileUseStepData], limit: int = 10) -> tuple[str, str]:
        if not trajectory:
            return "", ""

        recent_steps = trajectory[-limit:]
        recent_lines = []
        compressed_lines = []
        for step in recent_steps:
            action_text = step.action_s or (str(step.action) if step.action else "无动作")
            reasoning = step.action_desc or ""
            thought = step.thought or ""
            line_parts = [action_text]
            if reasoning:
                line_parts.append(f"###reasoning: {reasoning}")
            if thought:
                line_parts.append(f"###thought: {thought}")
            if step.memory:
                line_parts.append(f"###memory: {step.memory}")
            recent_lines.append(f"Step{step.step_idx + 1}: " + "".join(line_parts))
            if reasoning:
                compressed_lines.append(f"Step{step.step_idx + 1}: {reasoning}")

        return "\n".join(recent_lines), "\n".join(compressed_lines)

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        pixels = current_step.curr_env_state.pixels.copy()
        self.raw_size = (pixels.width, pixels.height)
        if self.max_pixels is not None:
            pixels = resize_image(pixels, self.max_pixels)
        self.resized_size = self.COORDINATE_SIZE
        self.parser.set_sizes(resized_size=self.resized_size, raw_size=self.raw_size)

        history_display = getattr(episodedata, "recent_history", "") or ""
        history_compressed = getattr(episodedata, "long_term_history", "") or ""
        if not history_display:
            history_display, fallback_compressed = self._format_history(trajectory[:-1], limit=self.num_histories or 10)
            if not history_compressed:
                history_compressed = fallback_compressed
        if not history_display:
            history_display = "无"
        if not history_compressed:
            history_compressed = episodedata.memory or "无"

        plan = current_step.plan or ""
        if current_step.sub_goal:
            plan = f"{plan}\n当前子目标：{current_step.sub_goal}".strip()
        if not plan:
            plan = "无"

        prompt = self.prompt.task_prompt.format(
            text=episodedata.goal,
            interactive_history="",
            plan=plan,
            history_display=history_display,
            history_compressed=history_compressed,
        )

        return [
            generate_message("system", self.prompt.system_prompt),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode_image_url(pixels)}},
                ],
            },
        ]

    def parse_response(self, content: str, size: tuple[float, float] = None, raw_size: tuple[float, float] = None):
        self.parser.set_sizes(resized_size=size or self.resized_size, raw_size=raw_size or self.raw_size)
        return self.parser.parse(content)
