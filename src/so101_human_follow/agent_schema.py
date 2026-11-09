from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


TOOL_NAME = "execute_interaction_plan"
PLAN_VERSION = "1.0"
MAX_ACTIONS = 12
MAX_REPLY_LENGTH = 180
DEFAULT_GREETING_COUNT = 2

SPEEDS = frozenset({"slow", "normal"})
INTENSITIES = frozenset({"small", "normal"})
GESTURES = frozenset({"nod", "shake_head", "wave", "wiggle"})
DANCE_STYLES = frozenset({"happy", "cute", "calm"})
RESPONSE_CATEGORIES = frozenset({"neutral", "happy", "curious", "cute", "encouraging"})

ALLOWED_ACTIONS = frozenset(
    {
        "perform_greeting",
        "open_gripper",
        "close_gripper",
        "enter_pet_pose",
        "leave_pet_pose",
        "return_to_neutral",
        "return_to_action_start",
        "perform_gesture",
        "perform_response_gesture",
        "perform_dance",
        "start_human_follow",
        "stop_human_follow",
        "wait",
    }
)

CONFIRMATION_ACTIONS = frozenset({"perform_dance", "start_human_follow"})


class AgentSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class AgentAction:
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class InteractionPlan:
    version: str
    reply: str
    actions: tuple[AgentAction, ...]
    requires_confirmation: bool


def _reject_constant(value: str) -> None:
    raise AgentSchemaError(f"Non-finite JSON number is not allowed: {value}")


def load_json_object(value: str | bytes | bytearray) -> dict[str, Any]:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise AgentSchemaError("Function arguments must be JSON text.")
    if len(value.encode("utf-8")) > 32_768:
        raise AgentSchemaError("Function arguments are too large.")
    try:
        parsed = json.loads(value, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AgentSchemaError("Function arguments are not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise AgentSchemaError("Function arguments must be a JSON object.")
    return parsed


def _exact_keys(value: Mapping[str, Any], required: set[str], optional: set[str] = set()) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise AgentSchemaError(f"Missing fields: {sorted(missing)}")
    if extra:
        raise AgentSchemaError(f"Unexpected fields: {sorted(extra)}")


def _enum(value: Any, allowed: frozenset[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise AgentSchemaError(f"{field} must be one of {sorted(allowed)}.")
    return value


def _integer(value: Any, low: int, high: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise AgentSchemaError(f"{field} must be an integer from {low} to {high}.")
    return value


def _number(value: Any, low: float, high: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentSchemaError(f"{field} must be a number.")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise AgentSchemaError(f"{field} must be between {low} and {high}.")
    return result


def _validate_action_arguments(name: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise AgentSchemaError(f"Arguments for {name} must be an object.")
    arguments = dict(raw)

    if name == "perform_greeting":
        _exact_keys(arguments, set(), {"count", "speed"})
        return {
            "count": _integer(arguments.get("count", DEFAULT_GREETING_COUNT), 1, 3, "count"),
            "speed": _enum(arguments.get("speed", "normal"), SPEEDS, "speed"),
        }
    if name in {
        "open_gripper",
        "close_gripper",
        "enter_pet_pose",
        "leave_pet_pose",
        "return_to_neutral",
        "return_to_action_start",
    }:
        _exact_keys(arguments, set(), {"speed"})
        return {"speed": _enum(arguments.get("speed", "slow"), SPEEDS, "speed")}
    if name == "perform_gesture":
        _exact_keys(arguments, {"gesture"}, {"count", "intensity", "speed"})
        return {
            "gesture": _enum(arguments["gesture"], GESTURES, "gesture"),
            "count": _integer(arguments.get("count", 1), 1, 3, "count"),
            "intensity": _enum(
                arguments.get("intensity", "normal"), INTENSITIES, "intensity"
            ),
            "speed": _enum(arguments.get("speed", "normal"), SPEEDS, "speed"),
        }
    if name == "perform_response_gesture":
        _exact_keys(arguments, {"category"}, {"speed"})
        return {
            "category": _enum(arguments["category"], RESPONSE_CATEGORIES, "category"),
            "speed": _enum(arguments.get("speed", "normal"), SPEEDS, "speed"),
        }
    if name == "perform_dance":
        _exact_keys(arguments, set(), {"style", "duration_seconds", "speed"})
        return {
            "style": _enum(arguments.get("style", "cute"), DANCE_STYLES, "style"),
            "duration_seconds": _number(
                arguments.get("duration_seconds", 5.0), 1.0, 10.0, "duration_seconds"
            ),
            "speed": _enum(arguments.get("speed", "slow"), SPEEDS, "speed"),
        }
    if name in {"start_human_follow", "stop_human_follow"}:
        _exact_keys(arguments, set())
        return {}
    if name == "wait":
        _exact_keys(arguments, {"seconds"})
        return {"seconds": _number(arguments["seconds"], 0.1, 10.0, "seconds")}
    raise AgentSchemaError(f"Action is not allowed: {name}")


def parse_interaction_plan(payload: Any) -> InteractionPlan:
    if not isinstance(payload, dict):
        raise AgentSchemaError("Interaction plan must be an object.")
    _exact_keys(
        payload,
        {"version", "reply", "actions", "requires_confirmation"},
    )
    if payload["version"] != PLAN_VERSION:
        raise AgentSchemaError(f"Unsupported plan version: {payload['version']!r}")
    reply = payload["reply"]
    if not isinstance(reply, str):
        raise AgentSchemaError("reply must be a string.")
    reply = reply.strip()
    if not reply:
        raise AgentSchemaError("reply cannot be empty.")
    if len(reply) > MAX_REPLY_LENGTH or any(ord(char) < 32 and char not in "\t\n" for char in reply):
        raise AgentSchemaError("reply is too long or contains control characters.")
    raw_actions = payload["actions"]
    if not isinstance(raw_actions, list) or len(raw_actions) > MAX_ACTIONS:
        raise AgentSchemaError(f"actions must be an array with at most {MAX_ACTIONS} items.")

    actions: list[AgentAction] = []
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, dict):
            raise AgentSchemaError(f"actions[{index}] must be an object.")
        _exact_keys(raw_action, {"name", "arguments"})
        name = raw_action["name"]
        if not isinstance(name, str) or name not in ALLOWED_ACTIONS:
            raise AgentSchemaError(f"actions[{index}].name is not allowed.")
        arguments = _validate_action_arguments(name, raw_action["arguments"])
        actions.append(AgentAction(name=name, arguments=arguments))

    requested_confirmation = payload["requires_confirmation"]
    if type(requested_confirmation) is not bool:
        raise AgentSchemaError("requires_confirmation must be a boolean.")
    requires_confirmation = requested_confirmation or any(
        action.name in CONFIRMATION_ACTIONS for action in actions
    )
    return InteractionPlan(
        version=PLAN_VERSION,
        reply=reply,
        actions=tuple(actions),
        requires_confirmation=requires_confirmation,
    )


def plan_to_dict(plan: InteractionPlan) -> dict[str, Any]:
    return {
        "version": plan.version,
        "reply": plan.reply,
        "actions": [
            {"name": action.name, "arguments": dict(action.arguments)}
            for action in plan.actions
        ],
        "requires_confirmation": plan.requires_confirmation,
    }


def apply_fixed_intent_rules(text: str, plan: InteractionPlan) -> InteractionPlan:
    """Make the small set of promised Chinese commands deterministic.

    The cloud model still supplies the natural-language reply, but it cannot
    reinterpret these product-level commands into another robot action.
    """

    compact = "".join(str(text).split()).strip("，。！!,.、；;")
    commands = {
        "你好": "perform_greeting",
        "你好呀": "perform_greeting",
        "你好啊": "perform_greeting",
        "嗨": "perform_greeting",
        "哈喽": "perform_greeting",
        "打个招呼": "perform_greeting",
        "和我打个招呼": "perform_greeting",
        "摸摸头": "enter_pet_pose",
        "低头让我摸": "enter_pet_pose",
        "摸好了": "leave_pet_pose",
        "抬起来": "leave_pet_pose",
        "起来吧": "leave_pet_pose",
        "打开夹爪": "open_gripper",
        "张开夹爪": "open_gripper",
        "关闭夹爪": "close_gripper",
        "合上夹爪": "close_gripper",
        "回到刚才的位置": "return_to_action_start",
        "回到动作前的位置": "return_to_action_start",
    }
    response_commands = {
        "认真听": "neutral",
        "认真听一下": "neutral",
        "歪头思考": "curious",
        "好奇一下": "curious",
        "开心一点": "happy",
        "开心回应": "happy",
        "兴奋一下": "happy",
        "撒个娇": "cute",
        "撒娇": "cute",
        "乖巧一点": "cute",
        "乖巧回应": "cute",
        "鼓励我一下": "encouraging",
        "鼓励一下": "encouraging",
        "给我加油": "encouraging",
    }
    # Speech normalization converts Chinese counts (including 一下) to digits.
    response_commands.update({
        phrase.replace("一下", "1下"): category
        for phrase, category in tuple(response_commands.items()) if "一下" in phrase
    })
    gesture_commands = {
        "点头1次": ("nod", 1), "点头一次": ("nod", 1),
        "点头2次": ("nod", 2), "点头两次": ("nod", 2), "点头二次": ("nod", 2),
        "点头3次": ("nod", 3), "点头三次": ("nod", 3),
        "左右挥1下": ("wave", 1), "左右挥一下": ("wave", 1),
    }
    commands.update({phrase: "perform_response_gesture" for phrase in response_commands})
    commands.update({phrase: "perform_gesture" for phrase in gesture_commands})
    command_pattern = re.compile(
        "|".join(re.escape(phrase) for phrase in sorted(commands, key=len, reverse=True))
    )
    separator_pattern = re.compile(r"(?:[，。！!,.、；;]+|然后|接着|之后|再)")
    actions: list[AgentAction] = []
    position = 0
    while position < len(compact):
        if actions:
            while separator := separator_pattern.match(compact, position):
                position = separator.end()
        # Match only a complete sequence of direct commands.  In particular,
        # quoted phrases, negations, questions and descriptive prose must not
        # trigger a forced motion merely because they contain "摸摸头".
        for prefix in ("请", "先"):
            if compact.startswith(prefix, position):
                position += len(prefix)
        command = command_pattern.match(compact, position)
        if command is None:
            return plan
        name = commands[command.group()]
        if name == "perform_greeting":
            arguments = {"count": DEFAULT_GREETING_COUNT, "speed": "normal"}
        elif name == "perform_response_gesture":
            arguments = {"category": response_commands[command.group()], "speed": "normal"}
        elif name == "perform_gesture":
            gesture, count = gesture_commands[command.group()]
            arguments = {"gesture": gesture, "count": count, "intensity": "normal", "speed": "normal"}
        else:
            arguments = {"speed": "slow"}
        actions.append(AgentAction(name, arguments))
        position = command.end()
        if compact.startswith("吧", position):
            position += 1
        if len(actions) > MAX_ACTIONS:
            return plan
    if not actions:
        return plan
    return InteractionPlan(
        version=PLAN_VERSION,
        reply="好呀。" if any(action.name == "enter_pet_pose" for action in actions) else plan.reply,
        actions=tuple(actions),
        requires_confirmation=False,
    )


def build_system_prompt(agent_name: str) -> str:
    quoted_name = json.dumps(agent_name, ensure_ascii=False)
    return f"""你是运行在用户本机的 SO-101 机械臂语音助手，你的名字是 {quoted_name}。

所有回答和动作都必须通过唯一工具 execute_interaction_plan 返回。只需要语言回答时，actions 必须为空。
你只能选择工具 Schema 允许的语义动作，绝对不能生成舵机 ID、角度、串口命令、EEPROM 操作、安全阈值或未定义动作。
你不能关闭、修改或绕过当前模式的本地保护、关节限制、速度限制、急停和最终验证。对话动作不依赖视觉；只有用户明确开启人体视觉跟随时才使用视觉跟随。
含义不明确、当前能力无法完成或可能危险时，用 reply 询问或说明，actions 留空。
不要因为每一句普通聊天都让机械臂运动。
reply 使用简短自然的口语中文，通常一到两句即可；像在和人聊天，不念工具名称、动作参数、舵机编号或技术术语。不要机械重复固定模板，不要堆叠“呀、啦、哦”等语气词，也不要使用波浪号表示拖长音。

固定交互规则：
1. 用户问候“你好、嗨、哈喽、打个招呼”时，回复自然友好的中文，并调用 perform_greeting，默认 count 为 2，只让夹爪张合两次；用户明确指定次数时遵照指定次数，支持 1–3 次。
2. 用户说“摸摸头、低头让我摸”时，先回复“好呀”一类自然短句，再调用 enter_pet_pose。本地会直接低头，停留两秒后自动恢复本次动作前的实测位置；不会先大幅抬头准备，也不等待“摸好了”口令。所有关节目标和幅度由本地决定，模型不得给角度。
3. 用户说“摸好了、抬起来、起来吧”时，调用 leave_pet_pose。它是从当前实测位置小幅抬头的独立动作；不要求之前执行过“摸摸头”，也不能要求用户先完成其他口令，不会强制回到连接时的姿势。
4. 用户要求停止跟随时调用 stop_human_follow；紧急“停止、别动”会由本地程序优先处理。
5. 跳舞和启动人体跟随必须 requires_confirmation=true。
6. 不要承诺机器人已经执行；只表达将要做什么，最终执行结果由本地程序决定。
7. 一句话包含多个明确动作时，按用户说的先后顺序放进同一工具的 actions 数组，不能只保留最后一个。例如“你好摸摸头”或“你好，摸摸头”必须依次包含 perform_greeting（count=2）和 enter_pet_pose，先完成夹爪张合两次，再低头停留两秒并自动回位。单独“摸摸头”不附加打招呼动作。
8. 引用、解释某个词语不等于要求执行；否定的动作不得加入 actions。用户明确要求只聊天、不执行动作时，actions 必须为空。
9. 用户明确要求“点头两次、左右挥一下”等动作时，用 perform_gesture 的正确 gesture 和 count；本地会完整完成往返并恢复动作前的位置，不得用随机通用动作替代。
10. 除了以上专用动作，在适合用肢体回应的普通对话中，可以偶尔调用一次 perform_response_gesture。category 只能选择 neutral（认真听/普通回应）、happy（开心/兴奋）、curious（好奇/思考）、cute（撒娇/乖巧）、encouraging（鼓励/加油）。例如“撒个娇”用 cute，“鼓励我一下”用 encouraging；不要为每一句普通聊天附加动作。
11. perform_response_gesture 只指定情绪类别，具体模板由本地随机选择，并避免最近重复；模板可能让一个、两个或三个关节同时运动。所有舵机目标和幅度、活动关节选择以及返回原位都由本地程序决定。不要输出关节编号、角度或自定义轨迹。
12. 专用动作优先；不能擅自在“你好、摸摸头、点头、挥手、停止”等明确指令后额外附加通用随机动作。只有用户明确同时提出多个要求时才按顺序组合。停止或执行失败后，不得要求机器人自动回位继续运动。
13. 用户明确说“回到刚才的位置、回到动作前的位置”时，调用 return_to_action_start；这是恢复最近动作开始前实测保存的位置。return_to_neutral 是连接时的起始姿势，不能代替此动作。若本地没有可用记录，应说明无法执行，不能猜位置。
14. 一句连续演示指令可以按顺序包含最多12个动作；必须将所有动作放入同一个计划，按顺序完整执行，不并发、不拆成多次工具调用。停止和回到刚才的位置属于单独的控制指令，不添加到演示序列中。

输入同时包含 original_text 和 normalized_text。优先按 normalized_text 理解，但如二者冲突或修正看起来不可靠，必须询问用户，不得猜测。"""


def interaction_tool_schema() -> dict[str, Any]:
    def action_variant(name: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "arguments"],
            "properties": {
                "name": {"type": "string", "enum": [name]},
                "arguments": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": required,
                    "properties": properties,
                },
            },
        }

    speed = {"type": "string", "enum": sorted(SPEEDS)}
    variants = [
        action_variant(
            "perform_greeting",
            {
                "count": {
                    "type": "integer", "minimum": 1, "maximum": 3,
                    "description": "夹爪张合次数，普通问候默认 2 次，明确指定时遵照用户要求。",
                },
                "speed": speed,
            },
            ["count", "speed"],
        )
    ]
    for name in (
        "open_gripper",
        "close_gripper",
        "enter_pet_pose",
        "leave_pet_pose",
        "return_to_neutral",
        "return_to_action_start",
    ):
        variants.append(action_variant(name, {"speed": speed}, ["speed"]))
    variants.extend(
        [
            action_variant(
                "perform_gesture",
                {
                    "gesture": {"type": "string", "enum": sorted(GESTURES)},
                    "count": {"type": "integer", "minimum": 1, "maximum": 3},
                    "intensity": {"type": "string", "enum": sorted(INTENSITIES)},
                    "speed": speed,
                },
                ["gesture", "count", "intensity", "speed"],
            ),
            action_variant(
                "perform_response_gesture",
                {
                    "category": {"type": "string", "enum": sorted(RESPONSE_CATEGORIES)},
                    "speed": speed,
                },
                ["category", "speed"],
            ),
            action_variant(
                "perform_dance",
                {
                    "style": {"type": "string", "enum": sorted(DANCE_STYLES)},
                    "duration_seconds": {"type": "number", "minimum": 1.0, "maximum": 10.0},
                    "speed": speed,
                },
                ["style", "duration_seconds", "speed"],
            ),
            action_variant("start_human_follow", {}, []),
            action_variant("stop_human_follow", {}, []),
            action_variant(
                "wait",
                {"seconds": {"type": "number", "minimum": 0.1, "maximum": 10.0}},
                ["seconds"],
            ),
        ]
    )
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "生成一条安全的语音回复和受白名单约束的本地机械臂交互计划。",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["version", "reply", "actions", "requires_confirmation"],
                "properties": {
                    "version": {"type": "string", "enum": [PLAN_VERSION]},
                    "reply": {"type": "string", "minLength": 1, "maxLength": MAX_REPLY_LENGTH},
                    "actions": {
                        "type": "array",
                        "maxItems": MAX_ACTIONS,
                        "items": {"anyOf": variants},
                    },
                    "requires_confirmation": {"type": "boolean"},
                },
            },
        },
    }
