from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import yaml

from mobile_use.agents.color_mobile.app_mapping import (
    build_app_to_package,
    build_package_to_app,
    format_app_names_for_planner,
)
from mobile_use.utils.vlm import VLMWrapper


@dataclass
class ColorMobileAppRetriever:
    """LLM-based app retriever for planner context."""

    vlm: VLMWrapper
    prompt_config: str = "color_mobile_app_retriever.yaml"
    enabled: bool = True
    chunk_size: int = 40
    max_apps: int = 30
    max_workers: int = 5

    def __post_init__(self):
        self.prompt = self._load_prompt(self.prompt_config)
        self.app_to_package, self.package_to_app = self._build_canonical_mapping()
        self.app_names = list(self.app_to_package.keys())

    @staticmethod
    def _load_prompt(prompt_config: str) -> dict:
        prompt_path = Path(__file__).parents[2] / "default_prompts" / prompt_config
        with open(prompt_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _build_canonical_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
        return build_app_to_package(), build_package_to_app()

    @staticmethod
    def _chunk(items: List[str], chunk_size: int) -> List[List[str]]:
        if chunk_size <= 0:
            return [items]
        return [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]

    @staticmethod
    def _format_app_chunk(app_names: List[str]) -> str:
        return "\n".join(f"{idx}. {app_name}" for idx, app_name in enumerate(app_names, start=1))

    @staticmethod
    def _extract_app_names(response_text: str, allowed_apps: List[str]) -> List[str]:
        selected = []
        for app_name in allowed_apps:
            if re.search(rf"(?<![\w]){re.escape(app_name)}(?![\w])", response_text):
                selected.append(app_name)
        return selected

    def _retrieve_chunk(self, query: str, app_chunk: List[str]) -> List[str]:
        user_prompt = self.prompt["user_prompt"].format(
            query=query,
            app_list=self._format_app_chunk(app_chunk),
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.prompt["system_prompt"]}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        response = self.vlm.predict(messages)
        response_text = response.choices[0].message.content.strip()
        return self._extract_app_names(response_text, app_chunk)

    def retrieve(self, query: str) -> List[str]:
        if not self.enabled or not query or self.vlm is None:
            return []

        chunks = self._chunk(self.app_names, self.chunk_size)
        selected = []
        seen_packages = set()

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
            future_to_chunk = {
                executor.submit(self._retrieve_chunk, query, chunk): chunk
                for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                try:
                    chunk_selected = future.result()
                except Exception:
                    continue
                for app_name in chunk_selected:
                    package_name = self.app_to_package.get(app_name)
                    if not package_name or package_name in seen_packages:
                        continue
                    seen_packages.add(package_name)
                    selected.append(app_name)
                    if self.max_apps and len(selected) >= self.max_apps:
                        return selected

        return selected

    def format_for_planner(self, query: str) -> str:
        selected = self.retrieve(query)
        return "、".join(selected) if selected else "未提供"

    @staticmethod
    def format_all_apps_for_planner() -> str:
        return format_app_names_for_planner()
