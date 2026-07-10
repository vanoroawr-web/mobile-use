import re
from typing import Dict, List

from mobile_use.agents.sub_agent import Planner
from mobile_use.schema.schema import MobileUseEpisodeData
from mobile_use.utils.utils import generate_message


class ColorMobilePlanner(Planner):
    """Chinese ColorMobileAgent planner."""

    @staticmethod
    def _split_value(line: str) -> str:
        if "：" in line:
            return line.split("：", 1)[1].strip()
        if ":" in line:
            return line.split(":", 1)[1].strip()
        return ""

    @staticmethod
    def _is_new_field(line: str) -> bool:
        normalized = line.strip()
        if not normalized:
            return False
        field_prefixes = (
            "意图",
            "改写后的query",
            "需打开的应用名",
            "query难度",
            "是否需要操作屏幕",
            "任务分解",
            "直接回答",
            "tips",
            "first_open_app",
        )
        if any(normalized.startswith(prefix) for prefix in field_prefixes):
            return True
        return bool(re.match(r"^(task|step|步骤)\d*\s*[:：]", normalized, flags=re.IGNORECASE))

    def _parse_planner_result(self, result_text: str) -> Dict:
        result = {
            "intent": "",
            "rewritten_query": "",
            "required_app": "",
            "query_difficulty": "",
            "needs_screen_operation": True,
            "direct_answer": "",
            "task_breakdown": [],
            "tips": "",
            "first_open_app": "",
        }

        collecting_direct_answer = False
        for original_line in result_text.splitlines():
            line = original_line.strip()

            if collecting_direct_answer and self._is_new_field(line):
                collecting_direct_answer = False

            if collecting_direct_answer:
                result["direct_answer"] = (
                    f"{result['direct_answer']}\n{original_line.rstrip()}".strip()
                    if result["direct_answer"]
                    else original_line.rstrip()
                )
                continue

            if not line:
                continue

            if line.startswith("意图"):
                result["intent"] = self._split_value(line)
            elif line.startswith("改写后的query"):
                result["rewritten_query"] = self._split_value(line)
            elif line.startswith("需打开的应用名") or line.startswith("需打开应用"):
                app_name = self._split_value(line)
                if app_name and app_name.lower() != "none" and app_name != "无":
                    result["required_app"] = app_name
            elif line.startswith("query难度") or "query难度" in line:
                difficulty = self._split_value(line) or line
                match = re.search(r"【([难易])】|([难易])", difficulty)
                result["query_difficulty"] = (match.group(1) or match.group(2)) if match else difficulty.strip("【】")
            elif line.startswith("是否需要操作屏幕") or "是否需要操作屏幕" in line:
                value = self._split_value(line) or line
                match = re.search(r"【([是否])】|([是否])", value)
                if match:
                    result["needs_screen_operation"] = (match.group(1) or match.group(2)) == "是"
                else:
                    result["needs_screen_operation"] = "否" not in value and "不需要" not in value
            elif line.startswith("直接回答"):
                answer = self._split_value(line)
                if answer:
                    result["direct_answer"] = answer
                collecting_direct_answer = True
            elif line.startswith("tips"):
                result["tips"] = self._split_value(line)
            elif line.startswith("first_open_app"):
                app_name = self._split_value(line)
                result["first_open_app"] = "" if app_name in ("无", "none", "None") else app_name
            elif re.match(r"^(task|step|步骤)\d*\s*[:：]", line, flags=re.IGNORECASE):
                task = self._split_value(line)
                task = re.sub(r"^(task|step|步骤)\d*\s*[:：]\s*", "", task, flags=re.IGNORECASE).strip()
                if task and task not in ("无", "直接回答"):
                    result["task_breakdown"].append(task)

        return result

    @staticmethod
    def _format_task_breakdown(tasks: List[str]) -> str:
        if not tasks:
            return "无"
        return "\n".join(f"task{idx}: {task}" for idx, task in enumerate(tasks, start=1))

    def _format_plan(self, parsed: Dict, raw_response: str) -> str:
        tasks = self._format_task_breakdown(parsed["task_breakdown"])
        lines = [
            f"意图：{parsed['intent'] or '未明确'}",
            f"改写后的query：{parsed['rewritten_query'] or '未改写'}",
            f"需打开的应用名：{parsed['required_app'] or '无'}",
            f"query难度：【{parsed['query_difficulty'] or '易'}】",
            f"是否需要操作屏幕：【{'是' if parsed['needs_screen_operation'] else '否'}】",
            "任务分解：",
            tasks,
        ]
        if parsed["direct_answer"]:
            lines.append(f"直接回答：{parsed['direct_answer']}")
        if parsed["tips"]:
            lines.append(f"tips：{parsed['tips']}")
        lines.append(f"first_open_app: {parsed['first_open_app'] or '无'}")
        plan = "\n".join(lines).strip()
        return plan if plan else raw_response.strip()

    def _format_operator_plan(self, parsed: Dict, raw_response: str) -> str:
        lines = []
        if parsed["intent"]:
            lines.append(f"意图：{parsed['intent']}")
        if parsed["rewritten_query"]:
            lines.append(f"改写后的query：{parsed['rewritten_query']}")
        if parsed["task_breakdown"]:
            lines.append("任务分解：")
            lines.append(self._format_task_breakdown(parsed["task_breakdown"]))
        elif raw_response.strip():
            lines.append(raw_response.strip())
        if parsed["tips"]:
            lines.append(f"tips：{parsed['tips']}")
        return "\n".join(lines).strip() or "无"

    @staticmethod
    def _current_subgoal(parsed: Dict) -> str:
        if not parsed["needs_screen_operation"]:
            return "Finished"
        if parsed["task_breakdown"]:
            return parsed["task_breakdown"][0]
        if parsed["rewritten_query"]:
            return parsed["rewritten_query"]
        return parsed["intent"] or "执行用户任务"

    @staticmethod
    def _get_installed_apps_context(episodedata: MobileUseEpisodeData) -> str:
        installed_apps = getattr(episodedata, "installed_apps", None)
        if not installed_apps:
            return "未提供"
        if isinstance(installed_apps, str):
            return installed_apps
        if isinstance(installed_apps, (list, tuple, set)):
            return "、".join(str(app) for app in installed_apps)
        return str(installed_apps)

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        prompt = self.prompt.task_prompt.format(
            query=episodedata.goal,
            related_install_apps_context=self._get_installed_apps_context(episodedata),
        )

        return [
            generate_message("system", self.prompt.system_prompt),
            generate_message("user", prompt),
        ]

    def parse_response(self, response: str):
        parsed = self._parse_planner_result(response)
        thought = parsed["intent"] or parsed["rewritten_query"] or response.strip()
        plan = self._format_operator_plan(parsed, response)
        current_subgoal = self._current_subgoal(parsed)
        return thought, plan, current_subgoal
