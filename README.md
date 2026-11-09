# SO-101 Human Follow

**当前版本：1.1（Python 包版本 `1.1.0`）。**

基于 LeRobot SO-101 从臂的实验项目，支持**电脑摄像头跟随右臂/右手**，以及**文字或语音对话驱动机械臂动作**。

项目通过 MediaPipe 提取右肩、右肘、右腕及右手关键点，计算肘部夹角、上臂姿态、手掌旋转和张合状态，再映射为 SO-101 舵机位置指令。人体跟随的视觉推理仍全部在本机完成，不依赖大语言模型或 ROS；可选的语音 Agent 才会调用用户自行配置的云端 OpenAI-compatible 模型。

> 当前是可运行的实验版本：ID3 肘部跟随和 ID6 夹爪张合已经在真机上跑通。不同机械臂、摄像头位置和使用者需要继续调整映射比例、方向与响应速度。

## 1.1 更新内容

- 增加文字/语音 Agent，通过 OpenAI-compatible Function Calling 生成本地白名单动作计划。
- 增加两次夹爪问候、低头停留2秒后回位、独立小幅抬头、点头、挥手和多关节随机回应，支持一句话顺序安排最多12个动作。
- 视频跟随与对话动作共享机械臂连接并互斥调度；相机故障不再阻断对话动作。
- 增加粉色/白色主题、功能弹窗、在线女声/男声和音频起播时启动动作的流程。
- 名称、API地址、模型与音色自动保存，Key仅存进程内存；兼容本地目录可写但不可重命名的权限。
- 改进动作到位反馈、回位、忙碌处理和后台诊断日志，并补充安装、参数与排查说明。

语音自然度、启动等待、实机动作方向和精确同步仍需继续优化，详见“当前限制”。

## 功能与使用入口

| 功能 | 输入 | 是否需要摄像头与人体中立姿势 | 是否需要云端模型 |
| --- | --- | --- | --- |
| 视频跟随 | 右臂、右手动作 | 需要；采集后 ARM | 不需要 |
| Agent 文字对话动作 | 页面输入自然语言 | 不需要 | 需要支持 Function Calling 的模型 |
| Agent 语音对话动作 | 电脑麦克风 | 不需要；还需浏览器支持语音识别 | 需要；在线回答音色也需要联网 |

首页连接从臂后，两种模式共用同一连接。“视频跟随”和“小助手对话”分别打开对应窗口；视觉跟随和 Agent 动作不能同时控制机械臂。

**本机当前使用地址：<http://127.0.0.1:7861/>。** 命令行和仓库启动脚本的默认端口仍为 `7860`；实际访问地址以启动时打印的 URL 为准。

## 当前效果与舵机映射

| ID | LeRobot 名称 | 机械臂作用 | 人体输入 | 当前行为 |
| --- | --- | --- | --- | --- |
| ID1 | `shoulder_pan` | 底座/肩部水平旋转 | 无 | 固定在连接时的位置 |
| ID2 | `shoulder_lift` | 肩部抬升 | 无 | 固定在连接时的位置 |
| ID3 | `elbow_flex` | 中部弯曲 | 右侧大臂与小臂夹角 | 已跑通 |
| ID4 | `wrist_flex` | 腕部俯仰 | 右侧上臂抬升/降低 | 实验中 |
| ID5 | `wrist_roll` | 腕部旋转 | 右手掌平面旋转 | 实验中 |
| ID6 | `gripper` | 夹爪开合 | 握拳/张手 | 已跑通，目标范围 0～100 |

ID1、ID2 在跟随期间保持不动；ID3～ID6 每个视觉通道独立更新。某个通道暂时识别不到时，对应舵机会保持上一位置。

界面中的 ID1～ID5 数值是 LeRobot 校准后的归一化位置，不等于严格的物理角度；ID6 使用 0～100 的夹爪位置范围。

## 技术栈与流程

- Python 3.10
- [LeRobot](https://github.com/huggingface/lerobot) `SO101Follower`
- OpenCV
- MediaPipe Tasks Pose Landmarker + Hand Landmarker
- Flask + 原生 HTML/CSS/JavaScript
- Feetech 串行舵机总线
- 可选 OpenAI-compatible Function Calling（仅语音 Agent）

```text
电脑摄像头
  → MediaPipe Pose / Hand
  → 肘部夹角、上臂抬升、手掌旋转、手掌开合
  → 映射、死区、平滑、限位、限速
  → LeRobot SO101Follower
  → COM 串口
  → 六个舵机
```

## 端口说明

项目里有两类不同的“端口”：

| 类型 | 默认值 | 用途 |
| --- | --- | --- |
| 本地网页端口 | `7860` | 浏览器访问 `http://127.0.0.1:7860/` |
| 本机本轮启动端口 | `7861` | 当前使用 `http://127.0.0.1:7861/`，由启动参数指定 |
| 机械臂串口 | `COM4` | Windows 与 SO-101 舵机控制板通信；按设备管理器实际编号修改 |

网页服务只监听 `127.0.0.1`，默认不能被局域网其他电脑直接访问。

### 如何修改机械臂串口

1. 打开 Windows“设备管理器”。
2. 展开“端口（COM 和 LPT）”。
3. 找到机械臂使用的 USB 串口，例如 `USB-Enhanced-SERIAL CH343 (COM4)`。
4. 打开项目根目录的 `config.json`，修改 `robot.port`：

```json
{
  "robot": {
    "port": "COM4",
    "id": "my_follower_arm"
  }
}
```

如果设备管理器显示 `COM7`，就改成 `"port": "COM7"`。`robot.id` 也必须改成执行 LeRobot 校准时使用的同一个 ID。

若要修改网页端口，把启动命令中的 `--http-port 7860` 改成其他未占用端口，例如 `--http-port 7861`，随后访问 `http://127.0.0.1:7861/`。

## 项目结构

```text
so101_human_follow/
├─ README.md                    # 使用、参数、实现与排查说明
├─ pyproject.toml               # Python 版本、依赖和命令入口
├─ config.example.json          # 可公开的配置模板
├─ config.json                  # 本机硬件/视觉配置，不提交 Git
├─ agent_preferences.json       # 名称、地址、模型、音色，不含 Key
├─ agent_preferences.backup.json # 上一份有效偏好，不含 Key
├─ safety_profile.json          # 工作范围、固定值与校准指纹
├─ logs/control.log             # 后端诊断日志
├─ models/                      # 视觉模型（不提交 Git）
├─ scripts/                     # 模型下载脚本
├─ src/so101_human_follow/
│  ├─ pose_detector.py          # Pose/Hand 推理
│  ├─ pose_math.py              # 关键点转控制特征
│  ├─ safety.py                 # 映射、平滑、限位与限速
│  ├─ robot_adapter.py          # LeRobot/串口适配
│  ├─ agent_schema.py           # 固定提示词、Function Calling Schema 与本地二次校验
│  ├─ agent_client.py           # OpenAI-compatible 模型请求
│  ├─ agent_service.py          # 云端请求工作线程，不接触串口
│  ├─ agent_actions.py          # 语义动作到安全关键帧的封装
│  ├─ speech_normalizer.py      # 转写词语/次数规范化与停止识别
│  ├─ speech_synthesis.py       # TTS、音色、文本清理与内存缓存
│  ├─ safe_range.py            # 本地工作范围档案
│  ├─ config.py                # 配置读取与校验
│  ├─ web_engine.py             # 控制状态机
│  ├─ web_app.py                # Flask 服务与 HTTP 接口
│  ├─ templates/index.html     # 页面、设置和功能窗口
│  └─ static/
│     ├─ app.js                # 机械臂、校准和视频跟随操作
│     ├─ agent-ui.js           # 对话、设置、语音播放与状态
│     ├─ style.css             # 基础样式
│     └─ agent-theme.css       # 粉色/白色主题
├─ start_web_preview.bat        # 仅视觉预览
└─ start_web_robot.bat          # 允许连接真机
```

## 安装

先按照 LeRobot 官方文档配置能够正常控制 SO-101 的 Python 3.10 环境。建议把本项目与 `lerobot` 仓库放在同一父目录，或者在当前环境中安装 LeRobot。

```powershell
git clone https://github.com/zhengxuan123456/so101-human-follow.git
cd so101-human-follow

# 激活已经能运行 LeRobot 的 Python 3.10 环境后：
python -m pip install -e ".[web]"
python -m pip install -r requirements-vision.txt
```

下载 MediaPipe 模型：

```powershell
python scripts\download_pose_model.py
python scripts\download_hand_model.py
```

复制配置模板：

```powershell
if (-not (Test-Path config.json)) {
    Copy-Item config.example.json config.json
}
```

修改 `config.json` 中的：

- `robot.port`：机械臂实际串口，例如 `COM4`。
- `robot.id`：必须与 LeRobot 校准时使用的机器人 ID 一致。
- `camera.index`：电脑摄像头通常为 `0`，请先通过预览确认。
- `pose.side`：默认跟随右臂，值为 `right`。

已有 `config.json` 时保留其中调好的参数。本机配置、运行日志、模型文件、偏好文件和安全范围被 `.gitignore` 排除；LeRobot 校准文件由 LeRobot 保存在用户缓存目录，不随本项目分发。

## 运行

### 1. 纯视觉预览

双击 `start_web_preview.bat`，或运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m so101_human_follow.web_app --config config.json --http-port 7860
```

打开 <http://127.0.0.1:7860/>。这一模式不会连接机械臂，也不会发送舵机指令。

确认相关关键点稳定：上臂抬升需要肩、肘，肘部夹角需要肩、肘、腕，手掌旋转和张合使用手部关键点。不同通道可分别有效，不要求完整全身同时入镜。

### 2. 允许网页连接真机

确认预览正常后，关闭预览服务并双击 `start_web_robot.bat`，或运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m so101_human_follow.web_app --config config.json --http-port 7860 --enable-robot
```

启动只会允许页面连接，不会自动连接、ARM 或运动。当前本机使用 `7861` 时，可运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m so101_human_follow.web_app --config config.json --http-port 7861 --enable-robot
```

按需要选择操作流程：

| 模式 | 操作顺序 |
| --- | --- |
| Agent 对话动作 | 保存 Agent 设置/补填 Key → 连接从臂 → 文字对话或开始语音 |
| 视频跟随 | 连接从臂 → 打开视频跟随 → 采集中立姿势 → ARM |

Agent 设置与机械臂连接两步的先后不作硬性要求，但发送动作前都必须就绪；人体中立姿势只用于视频跟随。

视频跟随的操作顺序：

1. 清空机械臂活动区域，并让实体电源开关保持触手可及。
2. 在网页中连接从臂；连接成功后仍为 `DISARMED`。
3. 采集中立姿势，等待进度完成。
4. 确认识别稳定后点击 ARM。
5. 先做一次很小的动作，确认各舵机方向正确。
6. 结束时先停止跟随，再托住机械臂并断开/释放扭矩。

基础舵机校准只在 LeRobot 校准不匹配时使用，不需要每次启动都重新校准。

重启服务会断开连接，并依配置释放扭矩；先托住机械臂。重启后刷新页面、补填 Key 并重新连接即可，其他已保存的 Agent 设置会恢复。

## 可选语音 Agent

新版网页包含“可爱粉色”和“简洁白色”两套皮肤。首页连接从臂后即可使用对话动作，不需要开启摄像头、采集中立姿势或ARM；只有使用“视频跟随”时才需要摄像头、中立姿势和ARM。旧的基础校准、连接诊断和六轴表折叠在高级区域中，后端能力没有删除。

点击右上角“Agent 设置”，填写：

- Agent 名字：语音回复时使用的自称。
- API Base URL：OpenAI-compatible 地址，例如 `https://api.example.com/v1`。
- API Key：只保存在当前 Python 后端进程内存，不写入配置文件、不回传页面、不记录日志；重启服务后需要重新填写。
- 模型名称：必须支持 Function Calling / Tool Calling。
- 语音：麦克风识别使用浏览器 SpeechRecognition；回答可选择“台妹女声·晓雨（自然语速）”和“小奶狗男声·年轻清朗云希”两个在线音色，也保留本机系统语音。女声使用`zh-TW-HsiaoYuNeural`，语速`+0%`、音高`+0Hz`，本机语音回退也使用1.0倍语速；男声参数保持不变。可以先点“试听所选音色”，实际听感以试听为准。在线音色通过 `edge-tts` 联网生成，不需要额外 Key，回复文字会发送给语音服务；生成失败时页面会明确提示已切换本机语音。

成功点击“保存设置”后，名字、API 地址、模型名称和音色会保存在本机 `agent_preferences.json`（Git 忽略），并在当前浏览器保留一份仅包含这四项的备份。本地另有 `agent_preferences.backup.json` 保存上一份有效设置；主文件写入中断时可恢复，不要求目录有删除或重命名权限。下次打开或重启服务后会自动填入，后端当前设置优先；不会自动连接模型或启动机械臂。API Key 不写入上述文件或浏览器存储，重启服务后只需补填 Key 再保存。

保存后可先使用文字输入验证。点击“开始语音 Agent”后，浏览器会持续监听，直到再次手动关闭；TTS 回答期间会暂停识别，回答结束后自动继续。浏览器不支持 SpeechRecognition 时仍可使用文字对话。

自然音色需要安装 `pip install "edge-tts>=7,<8"`（或安装项目的 `[web]` 依赖）。第一次合成可能需要几秒，相同回复的音频会在内存中短暂缓存，不保存录音文件。对话动作先准备音频，实际开始播放时才启动动作，并把音频实测时长交给后端；自动播放被浏览器阻止时，点击“播放并开始动作”后才执行。

统一时长控制函数是 `src/so101_human_follow/agent_actions.py` 的 `calculate_agent_motion_timing()`。它根据剩余关键帧的实际参与关节、行程、停顿和语音时长计算速度；在允许速度区间内使用计算值，区间外使用当前允许最高速度。ID3–ID5仍受原有单次步长限制，ID6使用其独立 `robot.max_relative_target.gripper` 限制；Agent目标更新频率提高到原控制频率的两倍（最高30Hz）。速度单位是LeRobot归一化位置/秒，不代表舵机硬件额定角速度。对话动作不受视觉状态减速或暂停；语音太短、硬件跟不上时，动作会延后完成，不跳步、不会取消参与关节的到位检查。以上调整不修改人体视觉跟随的映射参数。

固定交互：

- “你好”：ID6 夹爪张合两次，再恢复原位置；明确指定次数时支持1–3次。
- “摸摸头”：ID3、ID4从本次实测位置直接低头，幅度以校准范围的一半为目标、受剩余可用行程限制；停留2秒后自动回到本次动作前的位置。不先大幅抬头准备，也不需要再说“摸好了”。
- “摸好了 / 抬起来”：从当前实测位置小幅抬头，ID3、ID4分别使用8、10个归一化位置单位的位移，并受原有范围限制；这些数值不是角度。不要求之前执行过“摸摸头”，也不会强制摆到连接时的姿势。
- “点头两次”：ID3、ID4完成两次完整往返，结束回到这次点头前的实测位置。
- “左右挥一下”：ID5完成左右摆动并恢复这次挥手前的实测位置。
- “你好，摸摸头”：先完成夹爪两次张合，再低头停留2秒并回位；不会只保留后一个动作。
- “回到刚才的位置 / 回到动作前的位置”：明确要求恢复最近动作前保存的实测位置；不同于恢复连接姿势的`return_to_neutral`。
- “停止 / 别动 / 不要动”：本地直接停止，不等待云端模型。

通用回应动作使用`perform_response_gesture`，模型只能选择情绪类别，具体模板由本地动作库随机选择并尽量避免最近重复。可说“认真听”“歪头思考”“开心一点”“撒个娇”“鼓励我一下”。普通对话中，模型也可以在合适时偶尔选择一个回应动作，不会要求每句话都动。专用指令优先，不会被随机动作替换。

一条计划最多连续执行12个语义动作。完整演示可输入：“你好，然后打开夹爪，然后合上夹爪，然后摸摸头，然后抬起来，然后点头两次，然后左右挥一下，然后认真听，然后歪头思考，然后开心一点，然后撒个娇，然后鼓励我一下。”动作在同一计划内依次执行；“停止”和“回到刚才的位置”请单独发送，不加入演示序列。

| 类别 | 示例话语 | 表现 |
| --- | --- | --- |
| `neutral` | 认真听 | 认真听、轻微回应 |
| `curious` | 歪头思考、好奇一下 | 好奇、思考姿态 |
| `happy` | 开心一点、兴奋一下 | 活泼、开心回应 |
| `cute` | 撒个娇、乖巧一点 | 柔和、可爱回应 |
| `encouraging` | 鼓励我一下、给我加油 | 鼓励、肯定回应 |

每个通用动作会从ID3–ID6中选一个、两个或三个关节同时运动，ID1、ID2始终固定。一份计划开始前重新读取实测位置；单个回应、挥手和点头以本次起点作为原位并在结束后返回。复合计划的关键帧在开始时一次性展开，后续子动作起点由前一步计划位置递推，执行时持续读取反馈；尚未实现每个子动作边界都重新规划起点。回位同样计算行程、速度并检查到位。

语音播放完毕不代表机械臂已完成动作。上一个动作仍在执行或回位时，页面会提示等待，新输入保留且不会提交云端或排队执行；停止按钮始终可用，监听开启且未处于助手播音期间时也可说“停止”。若另一页面抢先开始了动作，本次尚未启动的计划只会被丢弃，不会连带取消正在执行的动作。

容错：STOP、断线、舵机不到位或回位失败时停止继续推进，不会为了自动回位再启动动作。程序保留最近动作前的位置记录；排除问题后，可明确说“回到动作前的位置”请求恢复，仍需通过连接、限位、速度和到位检查。没有可用记录时会拒绝恢复，不会猜角度。重新连接或重新校准后应以新会话的位置记录为准。日志会保留模板、关键帧、参与关节、目标和实测位置、失败原因及恢复结果，方便排查。

模型只能输出内置 JSON Schema 允许的语义动作，不能给舵机 ID、角度、串口、EEPROM 或安全阈值。结果还会在本地再次校验，并最终经过原有安全范围、单周期步长、ID1/ID2固定和串口单所有者机制。跳舞和启动人体跟随需要网页二次确认。

对话动作与视觉跟随分开：摄像头未启动、读帧失败或人物识别丢失不会阻止对话动作；关节限位、速度限制、实测到位检查和STOP仍然有效。视频跟随仍需要可靠的视觉输入。普通RGB摄像头只能用肩宽、掌宽的画面占比近似判断距离，不提供厘米级测距或工业碰撞检测；不要把对话模式当成视觉避障模式。

Agent、动作关键帧、视觉安全拦截和控制异常统一写入 `logs/control.log`（轮转保存）；日志不会记录 API Key。

## 实现方式

### 前后端与线程分工

页面由 Flask 模板生成，前端使用原生 JavaScript，通过 HTTP 读取状态和提交语义命令。机器人状态约每500ms轮询，Agent状态约每1000ms轮询，视频使用MJPEG流。

- `web_app.py`：检查本机访问、会话、同源请求与控制租约，提供HTTP接口。
- `agent_service.py`：独立工作线程请求模型，管理任务、确认、取消及非Key设置保存。
- `web_engine.py`：统一推进控制状态机，串行调度舵机读写；视觉跟随和Agent动作互斥。
- `pose_detector.py`：在独立视觉线程运行，避免摄像头推理拖慢对话动作发送。
- `robot_adapter.py`：封装LeRobot位置读取、目标发送、扭矩模式和基础校准。

相机读帧失败会停止依赖相机的视觉跟随；对话动作使用自己的连接、限位、反馈与停止检查。默认浏览器控制租约超时为2秒，丢失心跳后停止运动。

### 自然语言怎样变成动作

```text
文字 / 浏览器语音转写
  → speech_normalizer.py：词语、次数规范化，优先识别停止
  → agent_client.py：请求云端模型
  → execute_interaction_plan：返回结构化计划
  → agent_schema.py：白名单、类型、次数与完整结构校验
  → 固定直接指令规则兜底修正
  → READY：前端准备并播放回复音频
  → /api/agent/start：实际开播时提交音频时长
  → agent_actions.py：生成本地关键帧、活动关节与回位步骤
  → web_engine.py：限步长发送、实测到位、停留、下一步骤
  → robot_adapter.py → LeRobot → 串口舵机
```

对外工具只有 `execute_interaction_plan`，内部 `actions` 是允许的语义动作数组，例如：

```json
{
  "version": "1.0",
  "reply": "好呀，我先跟你打个招呼。",
  "actions": [
    {
      "name": "perform_greeting",
      "arguments": {"count": 2, "speed": "normal"}
    }
  ],
  "requires_confirmation": false
}
```

模型负责理解意思，本地代码决定关节目标。除本地停止外，当前正常对话先请求模型、校验返回计划，再应用固定直接指令规则，不是所有口令都绕过云端。

任务 `COMPLETED` 表示计划已派发，舵机进度要看 `agent_motion`；只有机械执行状态完成，才表示整段动作结束。每一步只判断参与动作的轴，身体关节反馈容差为3、夹爪为2个归一化单位，单关键帧未到位超时为15秒。这些是到位门槛，不会扩大工作范围。

### 关键接口

| 接口 | 用途 |
| --- | --- |
| `GET /api/status` | 机器人、识别、校准和机械动作状态 |
| `POST /api/connect`、`/api/disconnect` | 公共机械臂连接 |
| `POST /api/neutral`、`/api/arm` | 视频跟随的中立采集与启动 |
| `POST /api/stop` | 停止运动 |
| `GET/POST /api/agent/settings` | Agent设置读取与保存 |
| `POST /api/agent/test-connection` | 只测试模型工具调用结构 |
| `POST /api/agent/message` | 提交文字或语音转写 |
| `POST /api/agent/speech` | 合成回复音频 |
| `POST /api/agent/start` | 启动已准备的动作计划 |
| `POST /api/agent/cancel` | 取消交互，或只丢弃指定待执行任务 |
| `GET /api/agent/status`、`/api/agent/history` | 状态和历史展示 |

这些是应用内部接口；调用需页面会话、客户端ID及相应控制权，不能直接绕过页面验证操作串口。

## 配置参数

运行参数位于本机 `config.json`，公开模板为 `config.example.json`。

### `camera`：摄像头

| 参数 | 示例值 | 说明 |
| --- | ---: | --- |
| `index` | `0` | OpenCV 摄像头编号，当前默认电脑摄像头 |
| `width` / `height` | `640` / `480` | 采集分辨率 |
| `fps` | `30` | 目标帧率 |
| `rotation_degrees` | `0` | 画面旋转角度 |
| `flip_horizontal` / `flip_vertical` | `false` | 水平/垂直翻转 |

### `pose`：人体与手部识别

| 参数 | 示例值 | 说明 |
| --- | ---: | --- |
| `side` | `right` | 跟随右臂；可改为 `left` |
| `min_detection_confidence` | `0.5` | 人体检测阈值 |
| `min_presence_confidence` | `0.5` | 人体存在阈值 |
| `min_tracking_confidence` | `0.5` | 人体跟踪阈值 |
| `min_landmark_visibility` | `0.5` | 关键点最低可见度 |
| `hand_min_*_confidence` | `0.4` | 手部检测、存在和跟踪阈值 |
| `max_hand_wrist_distance` | `0.25` | 手部与姿态手腕的最大匹配距离 |
| `neutral_sample_count` | `30` | 中立采集的帧数进度参考；当前实际采集窗口为30秒，各通道可分开出现，结束时需有有效样本 |

### `robot`：机械臂与底层限速

| 参数 | 示例值 | 说明 |
| --- | ---: | --- |
| `port` | `COM4` | Windows 串口号 |
| `id` | `my_follower_arm` | LeRobot 校准 ID |
| `control_hz` | `10` | 舵机控制频率 |
| `max_relative_target` | 每轴字典 | LeRobot 根据实时位置执行的二次限速 |
| `disable_torque_on_disconnect` | `true` | 断开时释放扭矩；机械臂可能下坠 |

示例中 ID1～ID5 的底层单周期变化上限为 `6`，ID6 为 `25`，用于让夹爪更快完成大幅开合。

### `mapping`：人体到舵机映射

- `controlled_joints`：允许跟随的 ID3～ID6。
- `fixed_joints.names`：固定保持的 ID1、ID2。
- `fixed_joints.mode = startup`：固定在连接时读取到的位置。
- `gains`：动作放大比例；越大机械臂响应幅度越大。
- `signs`：动作方向，只允许 `1.0` 或 `-1.0`；方向相反时切换符号。
- `feature_deadbands`：过滤轻微视觉抖动的死区。

当前示例增益：ID3=`100`、ID4=`70`、ID5=`90`、ID6=`100`。ID6 使用专用的绝对开合映射，因此其 `gain` 目前不参与最终计算。

### `safety`：运动保护

| 参数 | 示例值 | 说明 |
| --- | ---: | --- |
| `ema_alpha` | `0.55` | 平滑系数；越大响应越快，也可能更抖 |
| `max_step_per_cycle` | `6` | ID3～ID5 单个控制周期最大变化 |
| `relative_limits` | 每轴字典 | 相对连接位置的最大变化 |
| `absolute_limits` | 每轴范围 | 自动控制的绝对范围 |
| `startup_body_abs_limit` | `200` | 允许连接时的宽启动范围，不等于自动运动范围 |
| `valid_frames_before_arm` | `3` | ARM 前至少连续有效的帧数 |
| `tracking_timeout_s` | `3` | 全部通道丢失多久后自动 DISARM |
| `max_person_center_jump` | `1.0` | 人物中心允许的最大跳变 |
| `skip_safe_range_calibration` | `true` | 当前实验模式跳过额外安全范围采集 |

`skip_safe_range_calibration=true` 便于当前原型测试，但不代表没有风险。公开使用者必须根据自己的机械臂调整 `absolute_limits`、`relative_limits`，并先进行小幅动作验证。

### ID6 夹爪专用映射

当前实现把手部开合得分放大到完整夹爪行程：

- 握拳得分约 `≤0.35`：ID6 目标为 `0`。
- 张手得分约 `≥0.55`：ID6 目标为 `100`。
- 中间区域：连续过渡。
- ID6 单周期步长上限：`25`。

如果不同光照或距离下夹爪幅度仍然偏小，可以结合 `logs/control.log` 中的 `hand_open` 遥测值调整 `safety.py` 的两个阈值。

## 常见问题与排查

| 现象 | 先检查什么 |
| --- | --- |
| 网页连接被拒绝 | 服务是否启动、访问端口是否与终端打印的URL一致 |
| 重启后页面提示会话无效 | 刷新控制台页面，再重新连接机械臂 |
| COM4打开失败或六个舵机都未找到 | 实际串口、电源、USB和舵机总线线缆；COM5/COM6蓝牙串口不一定是机械臂 |
| `calibration mismatch` | 检查日志中的EEPROM/JSON字段差异，再按高级基础校准流程处理 |
| Agent提示未连接 | 首页连接从臂；对话不需要先采集人体中立姿势 |
| 新指令提示上个动作执行中 | 等待动作/回位结束，或明确停止；不要仅以语音结束判断完成 |
| 设置保存失败 | 查看日志中的 `stage/errno/winerror`；当前已采用直接写入加备份，兼容目录可写但不可重命名的权限 |
| 语音慢或机械感明显 | 区分模型耗时、整段TTS生成、播放速度与机械动作耗时；当前音色自然度仍待改善 |
| 头部方向不符合预期 | 核对实际关节方向与校准；归一化数值增减不等于固定的“向前/向后” |

日志位于 `logs/control.log`，约5MiB轮转、保留5份历史。可按任务ID、关键帧编号、参与关节、目标/实测误差定位问题。日志包含用户指令和机械臂位置，分享前请检查内容。

## 当前限制

- 语音采用整段MP3合成后播放，尚未实现流式TTS；自然度、首句等待和动作同步仍需优化。
- 当前模型请求包含系统提示词和本次输入，没有把完整历史对话重放给模型；页面聊天记录不等于模型多轮记忆。
- 动作计划保存在任务内存中，不会默认把每次对话的JSON长期写到磁盘。
- `speed=slow/normal` 经过结构校验，但当前执行主要由本地步长限制和语音时长估算决定，不是直接设置舵机硬件转速。
- 一份复合计划的关键帧一次性展开，尚未在每个语义子动作边界重新采集起点进行规划。
- 对话模式没有视觉距离门槛；项目未实现可靠的厘米测距、完整逆运动学或碰撞检测。

## 开发检查

不连接摄像头和机械臂即可运行：

```powershell
python -m unittest discover -s tests -v
```

已有检查覆盖配置、映射、工作范围、适配器和Web状态。`test_agent_motion_fails_closed_when_visual_proximity_is_unusable` 仍保留旧的“对话必须有视觉”断言，与当前对话模式分离的行为不一致，尚待更新；完整运行时需结合这一已知过期用例判断结果。模拟检查不能代替真机方向、负载和回位验证。

## 安全提示

- 网页停止按钮不是物理急停；真正的急停是机械臂实体电源开关。
- 第一次运行必须从小幅动作开始，不能直接使用机械极限。
- 断开时可能释放扭矩并下坠，请提前托住或放低机械臂。
- 单目视觉会受到遮挡、光照、背景人物图像和摄像头角度影响。
- 本项目是动作模仿实验，不是精确逆运动学、碰撞检测或工业安全系统。

## 致谢

- 基于戊辰老师的 SO-101 项目思路进行二次开发。
- 机器人控制依赖 Hugging Face [LeRobot](https://github.com/huggingface/lerobot)。
- 视觉关键点识别使用 Google MediaPipe。

如果你有原项目的公开链接，欢迎提交 Issue 或 PR 补充更完整的上游出处。

## License

本项目以 [MIT License](LICENSE) 开源。
