from io import BytesIO
import base64
from unittest.mock import Mock, patch

from PIL import Image

from mobile_use.agents.base import Agent
from mobile_use.agents.color_mobile.app_mapping import (
    APP_MAPPING,
    COLOR_MOBILE_APP_MAPPING,
    build_app_to_package,
    build_alias_mapping,
    format_app_names_for_planner,
    resolve_app_package,
)
from mobile_use.agents.color_mobile.app_retriever import ColorMobileAppRetriever
from mobile_use.agents.color_mobile.memory import ColorMobileMemory
from mobile_use.agents.color_mobile.operator import ColorMobileOperator
from mobile_use.agents.color_mobile.planner import ColorMobilePlanner
from mobile_use.agents.color_mobile_agent import ColorMobileAgent
from mobile_use.schema.schema import Action, AgentState, EnvState, AgentStatus, MobileUseStepData


class FakeChoice:
    def __init__(self, content):
        self.message = Mock(content=content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


def test_color_mobile_agent_is_registered():
    assert Agent.by_name("ColorMobileAgent") is ColorMobileAgent


def test_color_mobile_app_mapping_dedupes_and_resolves_package():
    app_to_package = build_app_to_package({"哔哩哔哩": "tv.danmaku.bili", "B站": "tv.danmaku.bili", "美团": "pkg.meituan"})

    assert app_to_package == {"哔哩哔哩": "tv.danmaku.bili", "美团": "pkg.meituan"}
    assert APP_MAPPING[0]["name"] == "哔哩哔哩"
    assert "B站" in APP_MAPPING[0]["aliases"]
    assert COLOR_MOBILE_APP_MAPPING["B站"] == "tv.danmaku.bili"
    assert build_alias_mapping([{"name": "主应用", "package": "pkg.main", "aliases": ["别名"]}]) == {
        "主应用": "pkg.main",
        "别名": "pkg.main",
    }
    assert format_app_names_for_planner(["哔哩哔哩", "美团"]) == "哔哩哔哩、美团"
    assert resolve_app_package("B站") == "tv.danmaku.bili"
    assert resolve_app_package("美团") == "com.sankuai.meituan"
    assert resolve_app_package("com.example.app") == "com.example.app"


def test_color_mobile_app_retriever_merges_and_dedupes_model_results():
    vlm = Mock()
    vlm.predict.side_effect = [
        FakeResponse("相关应用：\n- 美团\n- 高德地图\n- 不存在应用"),
        FakeResponse("相关应用：\n- 百度地图\n- 美团"),
    ]
    retriever = ColorMobileAppRetriever(
        vlm=vlm,
        chunk_size=2,
        max_apps=10,
        max_workers=1,
    )
    retriever.app_to_package = {
        "美团": "com.sankuai.meituan",
        "高德地图": "com.autonavi.minimap",
        "百度地图": "com.baidu.BaiduMap",
        "大众点评": "com.dianping.v1",
    }
    retriever.package_to_app = {
        "com.sankuai.meituan": "美团",
        "com.autonavi.minimap": "高德地图",
        "com.baidu.BaiduMap": "百度地图",
        "com.dianping.v1": "大众点评",
    }
    retriever.app_names = ["美团", "高德地图", "百度地图", "大众点评"]

    assert retriever.retrieve("找附近餐厅并导航") == ["美团", "高德地图", "百度地图"]


def test_color_mobile_operator_parses_chinese_action():
    operator = ColorMobileOperator(
        Mock(
            vlm=Mock(
                model_dump=Mock(
                    return_value={
                        "model_name": "test-model",
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                    }
                )
            ),
            prompt_config="color_mobile_operator.yaml",
            num_histories=10,
            include_device_time=True,
            include_tips=True,
            include_a11y_tree=False,
            max_pixels=None,
            knowledge=None,
        )
    )

    thought, action, action_s, action_desc = operator.parse_response(
        "thought: 当前在搜索页###reasoning: 输入关键词###action: TYPE[100,200,咖啡]###REMEMBER: 已进入搜索页",
        size=(999, 999),
        raw_size=(2000, 3000),
    )

    assert thought == "当前在搜索页"
    assert action_desc == "输入关键词"
    assert action.name == "type"
    assert action.parameters["coordinate"] == (200, 601)
    assert action.parameters["text"] == "咖啡"
    assert "###REMEMBER:" in action_s


def test_color_mobile_operator_parses_call_user():
    operator = ColorMobileOperator(
        Mock(
            vlm=Mock(
                model_dump=Mock(
                    return_value={
                        "model_name": "test-model",
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                    }
                )
            ),
            prompt_config="color_mobile_operator.yaml",
            num_histories=10,
            include_device_time=True,
            include_tips=True,
            include_a11y_tree=False,
            max_pixels=None,
            knowledge=None,
        )
    )

    _, action, action_s, action_desc = operator.parse_response(
        "thought: 需要用户选择规格###reasoning: 询问规格###action: call_user[0#请选择颜色：黑色/白色]",
        size=(1000, 1000),
        raw_size=(1000, 1000),
    )

    assert action_desc == "询问规格"
    assert action.name == "call_user"
    assert action.parameters["text"] == "请选择颜色：黑色/白色"
    assert action_s == "call_user[0#请选择颜色：黑色/白色]"


def test_color_mobile_operator_maps_open_app_name_to_package():
    operator = ColorMobileOperator(
        Mock(
            vlm=Mock(
                model_dump=Mock(
                    return_value={
                        "model_name": "test-model",
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                    }
                )
            ),
            prompt_config="color_mobile_operator.yaml",
            num_histories=10,
            include_device_time=True,
            include_tips=True,
            include_a11y_tree=False,
            max_pixels=None,
            knowledge=None,
        )
    )

    _, action, action_s, _ = operator.parse_response(
        "thought: 需要打开美团###reasoning: 打开应用###action: open[美团]",
        size=(999, 999),
        raw_size=(999, 999),
    )

    assert action.name == "open"
    assert action.parameters["text"] == "com.sankuai.meituan"
    assert action_s == "open[美团]"


def test_color_mobile_planner_parses_source_style_output():
    planner = ColorMobilePlanner(
        Mock(
            vlm=Mock(
                model_dump=Mock(
                    return_value={
                        "model_name": "test-model",
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                    }
                )
            ),
            prompt_config="color_mobile_planner.yaml",
        )
    )

    thought, plan, current_subgoal = planner.parse_response(
        "\n".join(
            [
                "意图：在美团搜索咖啡",
                "改写后的query：美团 搜索 咖啡",
                "需打开的应用名：美团",
                "query难度：【易】",
                "是否需要操作屏幕：【是】",
                "任务分解：",
                "task1：打开美团并搜索咖啡",
                "tips：优先使用搜索框",
                "first_open_app: 美团",
            ]
        )
    )

    assert thought == "在美团搜索咖啡"
    assert "任务分解：" in plan
    assert "task1: 打开美团并搜索咖啡" in plan
    assert "需打开的应用名" not in plan
    assert "query难度" not in plan
    assert "是否需要操作屏幕" not in plan
    assert "first_open_app" not in plan
    assert current_subgoal == "打开美团并搜索咖啡"


def test_color_mobile_planner_builds_text_only_prompt():
    planner = ColorMobilePlanner(
        Mock(
            vlm=Mock(
                model_dump=Mock(
                    return_value={
                        "model_name": "test-model",
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                    }
                )
            ),
            prompt_config="color_mobile_planner.yaml",
        )
    )
    step = MobileUseStepData(
        step_idx=0,
        curr_env_state=EnvState(pixels=Image.new("RGB", (100, 100))),
    )
    episode = Mock(goal="打开美团搜索咖啡", trajectory=[step], installed_apps=["美团", "微信"])

    messages = planner.get_message(episode)
    user_text = "".join(content["text"] for content in messages[1]["content"] if content["type"] == "text")

    assert "用户任务：打开美团搜索咖啡" in user_text
    assert "已安装应用列表：美团、微信" in user_text
    assert "当前截图" not in user_text
    assert "screenshot" not in user_text.lower()
    assert "当前已有规划" not in user_text
    assert "上一个子目标" not in user_text
    assert all(content["type"] == "text" for content in messages[1]["content"])

    second_step = MobileUseStepData(
        step_idx=1,
        curr_env_state=EnvState(pixels=Image.new("RGB", (100, 100))),
        plan="旧规划",
        sub_goal="旧子目标",
    )
    episode.trajectory = [step, second_step]
    messages = planner.get_message(episode)
    user_text = "".join(content["text"] for content in messages[1]["content"] if content["type"] == "text")
    assert "当前已有规划" not in user_text
    assert "上一个子目标" not in user_text
    assert "旧规划" not in user_text
    assert "旧子目标" not in user_text


@patch("mobile_use.agents.base.Environment")
def test_color_mobile_agent_passes_retrieved_apps_to_planner(mock_environment):
    env = Mock()
    env.get_state.return_value = EnvState(pixels=Image.new("RGB", (100, 100)), device_time="now")
    mock_environment.return_value = env

    agent = Agent.from_params(
        {
            "type": "ColorMobileAgent",
            "vlm": {
                "model_name": "test-model",
                "api_key": "test-key",
                "base_url": "https://example.test/v1",
            },
            "env": {"serial_no": "device"},
            "max_steps": 1,
            "app_retriever": {"enabled": True},
            "planner": {
                "enabled": True,
                "prompt_config": "color_mobile_planner.yaml",
            },
            "operator": {
                "enabled": True,
                "name": "Operator",
                "prompt_config": "color_mobile_operator.yaml",
            },
        }
    )
    agent.app_retriever.format_for_planner = Mock(return_value="美团、高德地图")
    agent.planner.vlm.predict = Mock(
        return_value=FakeResponse(
            "\n".join(
                [
                    "意图：找附近餐厅",
                    "改写后的query：找附近餐厅",
                    "需打开的应用名：美团",
                    "query难度：【易】",
                    "是否需要操作屏幕：【是】",
                    "任务分解：",
                    "task1：查找附近餐厅",
                    "first_open_app: 美团",
                ]
            )
        )
    )
    agent.operator.vlm.predict = Mock(
        return_value=FakeResponse("thought: 已完成###reasoning: 任务完成###action: COMPLETE")
    )

    agent.reset("找附近餐厅")
    agent.step()

    planner_messages = agent.planner.vlm.predict.call_args.args[0]
    user_text = "".join(content["text"] for content in planner_messages[1]["content"] if content["type"] == "text")
    assert "已安装应用列表：美团、高德地图" in user_text


@patch("mobile_use.agents.base.Environment")
def test_color_mobile_agent_passes_all_apps_when_retriever_disabled(mock_environment):
    env = Mock()
    env.get_state.return_value = EnvState(pixels=Image.new("RGB", (100, 100)), device_time="now")
    mock_environment.return_value = env

    agent = Agent.from_params(
        {
            "type": "ColorMobileAgent",
            "vlm": {
                "model_name": "test-model",
                "api_key": "test-key",
                "base_url": "https://example.test/v1",
            },
            "env": {"serial_no": "device"},
            "max_steps": 1,
            "app_retriever": {"enabled": False},
            "planner": {
                "enabled": True,
                "prompt_config": "color_mobile_planner.yaml",
            },
            "operator": {
                "enabled": True,
                "name": "Operator",
                "prompt_config": "color_mobile_operator.yaml",
            },
        }
    )
    agent.planner.vlm.predict = Mock(
        return_value=FakeResponse(
            "\n".join(
                [
                    "意图：打开美团",
                    "改写后的query：打开美团",
                    "需打开的应用名：美团",
                    "query难度：【易】",
                    "是否需要操作屏幕：【是】",
                    "任务分解：",
                    "task1：打开美团",
                    "first_open_app: 美团",
                ]
            )
        )
    )
    agent.operator.vlm.predict = Mock(
        return_value=FakeResponse("thought: 已完成###reasoning: 任务完成###action: COMPLETE")
    )

    agent.reset("打开美团")
    agent.step()

    planner_messages = agent.planner.vlm.predict.call_args.args[0]
    user_text = "".join(content["text"] for content in planner_messages[1]["content"] if content["type"] == "text")
    assert "已安装应用列表：" in user_text
    assert "美团" in user_text
    assert "B站" not in user_text


@patch("mobile_use.agents.base.Environment")
def test_color_mobile_agent_plans_once_and_reuses_plan(mock_environment):
    env = Mock()
    env.get_state.return_value = EnvState(pixels=Image.new("RGB", (100, 100)), device_time="now")
    mock_environment.return_value = env

    agent = Agent.from_params(
        {
            "type": "ColorMobileAgent",
            "vlm": {
                "model_name": "test-model",
                "api_key": "test-key",
                "base_url": "https://example.test/v1",
            },
            "env": {"serial_no": "device"},
            "max_steps": 2,
            "planner": {
                "enabled": True,
                "prompt_config": "color_mobile_planner.yaml",
            },
            "operator": {
                "enabled": True,
                "name": "Operator",
                "prompt_config": "color_mobile_operator.yaml",
            },
        }
    )
    agent.planner.vlm.predict = Mock(
        return_value=FakeResponse(
            "\n".join(
                [
                    "意图：打开美团搜索咖啡",
                    "改写后的query：打开美团搜索咖啡",
                    "需打开的应用名：美团",
                    "query难度：【易】",
                    "是否需要操作屏幕：【是】",
                    "任务分解：",
                    "task1：在美团搜索咖啡",
                    "first_open_app: 美团",
                ]
            )
        )
    )
    agent.operator.vlm.predict = Mock(
        side_effect=[
            FakeResponse("thought: 第一步###reasoning: 打开应用###action: open[美团]"),
            FakeResponse("thought: 第二步###reasoning: 等待页面###action: WAIT"),
        ]
    )

    agent.reset("打开美团搜索咖啡")
    first_step = agent.step()
    agent.curr_step_idx = 1
    second_step = agent.step()

    agent.planner.vlm.predict.assert_called_once()
    assert second_step.plan == first_step.plan
    assert second_step.sub_goal == first_step.sub_goal


def test_color_mobile_operator_builds_prompt_shape():
    operator = ColorMobileOperator(
        Mock(
            vlm=Mock(
                model_dump=Mock(
                    return_value={
                        "model_name": "test-model",
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                    }
                )
            ),
            prompt_config="color_mobile_operator.yaml",
            num_histories=10,
            include_device_time=True,
            include_tips=True,
            include_a11y_tree=False,
            max_pixels=None,
            knowledge=None,
        )
    )
    step = MobileUseStepData(
        step_idx=0,
        curr_env_state=EnvState(pixels=Image.new("RGB", (100, 100))),
        plan="1. 打开美团\n2. 搜索咖啡",
        sub_goal="打开美团",
    )
    episode = Mock(goal="打开美团搜索咖啡", trajectory=[step], memory="")

    messages = operator.get_message(episode)
    system_text = "".join(
        content["text"]
        for content in messages[0]["content"]
        if content["type"] == "text"
    )
    user_text = "".join(
        content["text"]
        for content in messages[1]["content"]
        if content["type"] == "text"
    )

    assert "999 pixels and its height is 999 pixels" in system_text
    assert "{resized_width}" not in system_text
    assert "IMPORTANT GUIDELINE" not in system_text
    assert "###反思" not in system_text
    assert "你当前位于" not in user_text
    assert "### Background ###" in system_text
    assert "999 pixels and its height is 999 pixels" not in user_text
    assert "The user's instruction is: 打开美团搜索咖啡." in user_text
    assert "以下内容是规划智能体的输出，请重点参考：" in user_text
    assert "### execution history ###" in user_text
    assert "【近期操作历史】" in user_text
    assert "【长期操作历史】" in user_text
    assert "### 当前截图 ###" not in user_text
    assert any(content["type"] == "image_url" for content in messages[1]["content"])

    image_content = next(content for content in messages[1]["content"] if content["type"] == "image_url")
    image_b64 = image_content["image_url"]["url"].split(",", 1)[1]
    sent_image = Image.open(BytesIO(base64.b64decode(image_b64)))
    assert sent_image.size == (100, 100)
    assert operator.resized_size == (999, 999)
    assert operator.raw_size == (100, 100)


def test_color_mobile_operator_max_pixels_resizes_image_proportionally():
    operator = ColorMobileOperator(
        Mock(
            vlm=Mock(
                model_dump=Mock(
                    return_value={
                        "model_name": "test-model",
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                    }
                )
            ),
            prompt_config="color_mobile_operator.yaml",
            num_histories=10,
            include_device_time=True,
            include_tips=True,
            include_a11y_tree=False,
            max_pixels=1024 * 1024,
            knowledge=None,
        )
    )
    step = MobileUseStepData(
        step_idx=0,
        curr_env_state=EnvState(pixels=Image.new("RGB", (2000, 1000))),
        plan="无",
    )
    episode = Mock(goal="测试", trajectory=[step], memory="")

    messages = operator.get_message(episode)
    image_content = next(content for content in messages[1]["content"] if content["type"] == "image_url")
    image_b64 = image_content["image_url"]["url"].split(",", 1)[1]
    sent_image = Image.open(BytesIO(base64.b64decode(image_b64)))

    assert sent_image.size == (1448, 724)
    assert sent_image.width * sent_image.height <= 1024 * 1024
    assert sent_image.width / sent_image.height == 2
    assert operator.resized_size == (999, 999)
    assert operator.raw_size == (2000, 1000)


def test_color_mobile_memory_keeps_recent_notes():
    memory = ColorMobileMemory(max_history_steps=2)
    step = MobileUseStepData(
        step_idx=0,
        curr_env_state=EnvState(pixels=Image.new("RGB", (10, 10))),
        action=Action(name="click", parameters={"coordinate": (1, 1)}),
        action_desc="Click the button",
    )

    assert memory.update(step) == "Step 1: Click the button"


def test_color_mobile_memory_compresses_long_term_history():
    memory = ColorMobileMemory(max_history_steps=4, compress_every_steps=2, recent_steps_after_compress=2)
    trajectory = []
    for idx in range(4):
        step = MobileUseStepData(
            step_idx=idx,
            curr_env_state=EnvState(pixels=Image.new("RGB", (10, 10))),
            action=Action(name="click", parameters={"coordinate": (idx, idx)}),
            action_s=f"CLICK[{idx},{idx}]",
            action_desc=f"第{idx + 1}步",
        )
        trajectory.append(step)

    assert memory.maybe_compress(trajectory[:3]) == ""

    long_term = memory.maybe_compress(trajectory)

    assert "Step1" in long_term
    assert "Step2" in long_term
    assert "Step3" not in long_term
    assert "Step4" not in long_term
    assert memory._last_compressed_step == 2

    recent_history = memory.get_recent_history(trajectory)
    assert "Step3" in recent_history
    assert "Step4" in recent_history
    assert "Step2" not in recent_history


def test_color_mobile_memory_compression_title_covers_accumulated_steps():
    memory = ColorMobileMemory(max_history_steps=4, compress_every_steps=2, recent_steps_after_compress=2)
    trajectory = []
    for idx in range(6):
        trajectory.append(
            MobileUseStepData(
                step_idx=idx,
                curr_env_state=EnvState(pixels=Image.new("RGB", (10, 10))),
                action=Action(name="click", parameters={"coordinate": (idx, idx)}),
                action_s=f"CLICK[{idx},{idx}]",
                action_desc=f"第{idx + 1}步",
            )
        )

    first_long_term = memory.maybe_compress(trajectory[:4])
    second_long_term = memory.maybe_compress(trajectory[:6])

    assert "对step1~step2步的总结：" in first_long_term
    assert "对step1~step4步的总结：" in second_long_term
    assert "对step3~step4步的总结：" not in second_long_term
    assert "Step1" in second_long_term
    assert "Step2" in second_long_term
    assert "Step3" in second_long_term
    assert "Step4" in second_long_term
    assert "Step5" not in second_long_term
    assert "Step6" not in second_long_term
    assert memory._last_compressed_step == 4

    recent_history = memory.get_recent_history(trajectory)
    assert "Step5" in recent_history
    assert "Step6" in recent_history
    assert "Step4" not in recent_history


def test_color_mobile_memory_recent_history_matches_execution_format():
    memory = ColorMobileMemory(max_history_steps=2)
    trajectory = [
        MobileUseStepData(
            step_idx=0,
            curr_env_state=EnvState(pixels=Image.new("RGB", (10, 10))),
            action=Action(name="click", parameters={"coordinate": (10, 20)}),
            action_s="CLICK[10,20]###REMEMBER: 已看到咖啡入口",
            action_desc="点击入口",
            thought="当前在首页",
        )
    ]

    recent_history = memory.get_recent_history(trajectory)

    assert "Step1：CLICK###reasoning: 点击入口###thought: 当前在首页" == recent_history
    assert "10,20" not in recent_history
    assert "REMEMBER" not in recent_history


def test_color_mobile_memory_uses_model_for_compression():
    vlm = Mock()
    vlm.predict.return_value = FakeResponse("已完成打开应用，接下来需要搜索咖啡")
    memory = ColorMobileMemory(
        max_history_steps=2,
        compress_every_steps=2,
        recent_steps_after_compress=0,
        vlm=vlm,
    )
    trajectory = [
        MobileUseStepData(
            step_idx=0,
            curr_env_state=EnvState(pixels=Image.new("RGB", (10, 10))),
            action=Action(name="open", parameters={"text": "美团"}),
            action_s="open[美团]",
            action_desc="打开美团",
        ),
        MobileUseStepData(
            step_idx=1,
            curr_env_state=EnvState(pixels=Image.new("RGB", (10, 10))),
            action=Action(name="click", parameters={"coordinate": (1, 1)}),
            action_s="CLICK[1,1]",
            action_desc="点击搜索框",
        ),
    ]

    long_term = memory.maybe_compress(trajectory, goal="打开美团搜索咖啡")

    assert long_term == "对step1~step2步的总结：\n已完成打开应用，接下来需要搜索咖啡"
    vlm.predict.assert_called_once()


@patch("mobile_use.agents.base.Environment")
def test_color_mobile_agent_step_finishes_on_terminate(mock_environment):
    env = Mock()
    env.get_state.return_value = EnvState(pixels=Image.new("RGB", (100, 100)), device_time="now")
    mock_environment.return_value = env

    agent = Agent.from_params(
        {
            "type": "ColorMobileAgent",
            "vlm": {
                "model_name": "test-model",
                "api_key": "test-key",
                "base_url": "https://example.test/v1",
            },
            "env": {"serial_no": "device"},
            "max_steps": 1,
            "planner": {"enabled": False},
            "operator": {
                "enabled": True,
                "name": "Operator",
                "prompt_config": "color_mobile_operator.yaml",
            },
        }
    )
    agent.operator.vlm.predict = Mock(
        return_value=FakeResponse(
            "thought: 任务已经完成###reasoning: 任务完成###action: COMPLETE"
        )
    )

    agent.reset("finish the task")
    step_data = agent.step()

    assert step_data.action.name == "terminate"
    assert agent.status == AgentStatus.FINISHED
    env.execute_action.assert_not_called()


@patch("mobile_use.agents.base.Environment")
def test_color_mobile_agent_step_stops_on_call_user(mock_environment):
    env = Mock()
    env.get_state.return_value = EnvState(pixels=Image.new("RGB", (100, 100)), device_time="now")
    mock_environment.return_value = env

    agent = Agent.from_params(
        {
            "type": "ColorMobileAgent",
            "vlm": {
                "model_name": "test-model",
                "api_key": "test-key",
                "base_url": "https://example.test/v1",
            },
            "env": {"serial_no": "device"},
            "max_steps": 1,
            "planner": {"enabled": False},
            "operator": {
                "enabled": True,
                "name": "Operator",
                "prompt_config": "color_mobile_operator.yaml",
            },
        }
    )
    agent.operator.vlm.predict = Mock(
        return_value=FakeResponse(
            "thought: 需要用户选择###reasoning: 询问用户###action: call_user[0#请选择规格]"
        )
    )

    agent.reset("购买咖啡")
    step_data = agent.step()

    assert step_data.action.name == "call_user"
    assert agent.state == AgentState.CALLUSER
    env.execute_action.assert_not_called()
