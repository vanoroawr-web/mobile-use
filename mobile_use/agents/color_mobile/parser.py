import re
from typing import Optional

from mobile_use.agents.color_mobile.app_mapping import resolve_app_package
from mobile_use.schema.schema import Action


class ColorMobileActionParser:
    """Parse ColorMobileAgent action strings."""

    def __init__(self):
        self.resized_size = None
        self.raw_size = None

    def set_sizes(self, resized_size=None, raw_size=None) -> None:
        if resized_size is not None:
            self.resized_size = resized_size
        if raw_size is not None:
            self.raw_size = raw_size

    def _scale_coordinate(self, x: int, y: int) -> tuple[int, int]:
        if not self.resized_size or not self.raw_size:
            return x, y
        resized_width, resized_height = self.resized_size
        raw_width, raw_height = self.raw_size
        return round(x / resized_width * raw_width), round(y / resized_height * raw_height)

    @staticmethod
    def extract_section(content: str, name: str, end_names: tuple[str, ...]) -> Optional[str]:
        end_pattern = "|".join(re.escape(end) for end in end_names)
        match = re.search(
            rf"{re.escape(name)}\s*(.*?)(?={end_pattern}|$)",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_remember(content: str) -> Optional[str]:
        match = re.search(r"###REMEMBER\s*:\s*(.+?)(?=###|$)", content, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_action(content: str) -> str:
        match = re.search(r"###action\s*:\s*(.+?)(?=###|$)", content, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError("Cannot extract ###action.")
        return match.group(1).strip()

    @staticmethod
    def _parse_bracket(action_text: str) -> tuple[str, Optional[str]]:
        match = re.match(r"([A-Za-z_]+)\s*(?:\[(.*)\])?\s*$", action_text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Invalid action: {action_text}")
        return match.group(1).strip().upper(), match.group(2)

    @staticmethod
    def _split_csv_params(params: str, expected_prefix_count: int) -> list[str]:
        parts = params.split(",", expected_prefix_count)
        if len(parts) < expected_prefix_count + 1:
            raise ValueError(f"Invalid action parameters: {params}")
        return [part.strip() for part in parts]

    def parse(self, content: str):
        thought = self.extract_section(content, "thought:", ("###reasoning:", "###action:", "###REMEMBER:"))
        action_desc = self.extract_section(content, "###reasoning:", ("###action:", "###REMEMBER:"))
        action_text = self._extract_action(content)
        remember = self.extract_remember(content)

        name, params = self._parse_bracket(action_text)
        if name == "CLICK":
            x_s, y_s = self._split_csv_params(params or "", 1)
            action = Action(name="click", parameters={"coordinate": self._scale_coordinate(int(x_s), int(y_s))})
        elif name == "DOUBLE_CLICK":
            x_s, y_s = self._split_csv_params(params or "", 1)
            action = Action(name="click", parameters={"coordinate": self._scale_coordinate(int(x_s), int(y_s))})
        elif name == "LONG_PRESS":
            x_s, y_s = self._split_csv_params(params or "", 1)
            action = Action(name="long_press", parameters={"coordinate": self._scale_coordinate(int(x_s), int(y_s))})
        elif name == "TYPE":
            x_s, y_s, text = self._split_csv_params(params or "", 2)
            action = Action(
                name="type",
                parameters={"coordinate": self._scale_coordinate(int(x_s), int(y_s)), "text": text},
            )
        elif name == "SWIPE":
            x1_s, y1_s, x2_s, y2_s = self._split_csv_params(params or "", 3)
            action = Action(
                name="swipe",
                parameters={
                    "coordinate": self._scale_coordinate(int(x1_s), int(y1_s)),
                    "coordinate2": self._scale_coordinate(int(x2_s), int(y2_s)),
                },
            )
        elif name == "WAIT":
            action = Action(name="wait", parameters={"time": 2})
        elif name == "OPEN":
            action = Action(name="open", parameters={"text": resolve_app_package(params or "")})
        elif name == "CALL_USER":
            text = (params or "").strip()
            if "#" in text:
                _, text = text.split("#", 1)
            action = Action(name="call_user", parameters={"text": text.strip()})
        elif name == "SYSTEM_BUTTON":
            button = (params or "back").strip().lower()
            button_map = {"back": "Back", "home": "Home", "menu": "Menu", "enter": "Enter"}
            action = Action(name="system_button", parameters={"button": button_map.get(button, "Back")})
        elif name == "ANSWER":
            action = Action(name="answer", parameters={"text": (params or "").strip()})
        elif name == "COMPLETE":
            action = Action(name="terminate", parameters={"status": "success"})
        else:
            raise ValueError(f"Unsupported action: {name}")

        action_s = action_text
        if remember:
            action_s = f"{action_s}###REMEMBER: {remember}"
        return thought, action, action_s, action_desc
