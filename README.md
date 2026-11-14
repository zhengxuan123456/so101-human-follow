# SO-101 Human Follow

**版本 1.2.0** · Windows 本地网页控制台 · [架构与源码指南](ARCHITECTURE.md)

用电脑摄像头让 SO-101 跟随手臂和手部动作，也可以用文字、语音或已录好的动作控制机械臂。网页内置六轴 MuJoCo 仿真、动作编排和主从臂示教，便于先预览，再连接真机测试。项目基于 Python、Flask、LeRobot 和 MediaPipe 运行，无需 ROS 或 MCP。

## 1.2 更新了什么

| 更新 | 使用时有什么变化 |
| --- | --- |
| 本地指令 | 「打开夹爪」「点头两次」等明确指令在本机转成动作计划，无需模型 Key。明确单动作可以先派发，语音并行准备；复杂对话和需要结合上一轮的插话依然可以走云端 |
| 返回原位 | 说「返回原位」或「回到连接时的位置」，恢复本次连接从臂时实测的六轴姿势。每次连接重新记录。目标超出当前工作范围时会提示 |
| 连续回放 | 支持按从臂录制的实际时间轨迹连续控制，不必在每个采样点等舵机到位。超速片段自动延长，持续跟不上目标时停止 |
| 主从臂示教 | 网页输入两个串口，可单录主臂或同步跟随。命名、介绍、主臂轨迹和从臂实测轨迹保存在同一个 JSON，可从轨迹提取可编辑的动作步骤 |
| 组合动作 | 把保存的动作按顺序组合，为每项选择关键帧或从臂轨迹，设置重复和停留，整套结束可选回位 |
| 本地相机与日志 | 按设备名称连接或断开相机。后台记录规划、语音、串口周期、目标及编码器误差，方便定位延迟和卡顿 |

**升级已有安装**时，保留自己的 `config.json`、LeRobot 校准和 `action_library/`，再更新代码、补齐依赖并重新启动控制台。已有关键帧动作仍可载入编辑，旧录制缺少从臂校准指纹时可在仿真中预览。要使用真机连续回放，请重新做一次同步示教。

## 快速开始

### 1. 安装与配置

使用 Python 3.10，先准备已经能运行 SO-101 的 [LeRobot 环境](https://huggingface.co/docs/lerobot/main/en/so101)。可把项目与 `lerobot` 源码目录放在同一父目录，或使用已安装的 LeRobot 包。

```powershell
git clone https://github.com/zhengxuan123456/so101-human-follow.git
Set-Location so101-human-follow
python -m pip install -e ".[web,simulation]"
python -m pip install -r requirements-vision.txt
python scripts\download_pose_model.py
python scripts\download_hand_model.py
if (-not (Test-Path config.json)) {
    Copy-Item config.example.json config.json
}
```

仅使用对话或视频时可把 `.[web,simulation]` 换成 `.[web]`。使用 MuJoCo 时还需要兼容 SO-101 的 MJCF 场景和网格。默认读取项目旁边的 `mujoco_sim\scene.xml`；也可在启动前设置 `$env:SO101_MUJOCO_MODEL` 为场景 XML 的完整路径。

### 2. 打开本地网页

- **预览与仿真**：运行 `start_web_preview.bat`。
- **允许连接真机**：运行 `start_web_robot.bat`，按提示确认。服务启动后由你在网页上连接从臂。

默认地址为 [http://127.0.0.1:7860/](http://127.0.0.1:7860/)。也能从项目根目录运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m so101_human_follow.web_app --config config.json --http-port 7860 --enable-robot
```

| 要设置什么 | 在哪里改 |
| --- | --- |
| 网页端口 `7860` | 启动命令的 `--http-port`，例如改成 `7861`，随后浏览器访问对应地址 |
| 从臂串口，例如 `COM4` | 首页「从臂串口」输入框；默认值在 `config.json` 的 `robot.port` |
| 主臂串口，例如 `COM3` | 「六轴仿真与动作编辑」中的「主从臂示教录制」 |
| LeRobot 机器 ID | `config.json` 的 `robot.id`，必须和从臂校准时使用的 ID 一致 |
| 相机 | 「视频跟随」中按 Windows 设备名称选择并连接，使用结束可主动断开 |

串口请以 Windows「设备管理器 → 端口（COM 和 LPT）」中的 USB 串口为准。双臂同步时，主臂和从臂需使用不同串口。

## 对话口令

连接从臂并准备好工作范围后，可直接在「小助手对话」里打字或使用浏览器语音识别。

| 口令 | 动作 |
| --- | --- |
| 你好 | ID6 夹爪张合两次，其他关节可小幅配合，结束回位 |
| 打开夹爪 / 合上夹爪 | 张开或合拢 ID6，保留指定开合状态 |
| 摸摸头 | 以 ID3、ID4 为主做低头动作，停留约两秒后回位 |
| 抬起来 / 摸好了 | 从当前姿势独立小幅抬头 |
| 点头两次 | 按指定次数点头。固定口令支持一至三次 |
| 左右挥一下 | 以 ID5 为主左右摆动，随后回位 |
| 认真听 / 歪头思考 / 开心一点 / 撒个娇 / 鼓励我一下 | 从对应类别选一个回应动作，做完回位 |
| 返回原位 | 恢复本次连接从臂时记录的六轴位置 |
| 回到刚才的位置 | 恢复最近一次动作开始前记录的位置，请单独发送 |
| 停止 / 别动 | 优先终止当前动作，保持停下时的位置 |

可以连说「你好，然后打开夹爪，再合上夹爪，接着点头两次，然后返回原位」。本地规则最多解析 12 项清晰指令。明确单动作可直接启动，多动作和云端动作可以等语音实际开播后启动。

开放对话、根据名称和介绍选择自定义动作、通用跳舞等使用支持 Function Calling 的 OpenAI-compatible 模型。「Agent 设置」里填写 Base URL、模型名称和自己的 API Key。名称、模型、地址、语言和音色保存在本机，Key 仅在服务进程内存中保留。浏览器语音识别和在线音色的联网需求取决于所选服务。文字固定口令可在没有模型 Key 的情况下使用。

## 视频跟随、示教和仿真

### 视频跟随

1. 在首页连接从臂，确认工作范围和摆放安全。
2. 打开「视频跟随」，选择相机并连接。
3. 采集中立姿势，确认右臂和手部的有效通道，再点击 ARM。
4. 小幅测试各轴方向。使用结束后先停止，再断开相机与从臂。

首次设置、换设备或遇到校准不匹配时，按网页的基础校准流程处理。基础舵机校准负责编码器零点和范围，人体中立姿势负责视觉映射。Agent 和示教各自有独立的启动条件，不需要采集人体中立姿势。

| 舵机 | 关节名 | 视频跟随用途 | 公开模板归一化范围 |
| --- | --- | --- | --- |
| ID1 | `shoulder_pan.pos` | 固定底座水平位置 | `-70 .. 70` |
| ID2 | `shoulder_lift.pos` | 固定肩部位置 | `-70 .. 70` |
| ID3 | `elbow_flex.pos` | 右侧大臂与小臂夹角 | `-70 .. 70` |
| ID4 | `wrist_flex.pos` | 右侧上臂抬升或降低 | `-100 .. 100` |
| ID5 | `wrist_roll.pos` | 手掌旋转 | `-100 .. 100` |
| ID6 | `gripper.pos` | 握拳与张手 | `0 .. 100` |

动作编辑、Agent 和示教可使用六轴。表中的位置值是 LeRobot 归一化单位，跟 EEPROM 读数和物理角度分别保存。实际工作范围由本机配置和校准限制决定。

### 主从臂示教与动作库

1. 打开「六轴仿真与动作编辑」，填写主从臂串口。用已有有效校准或现场完成校准。
2. 填动作名称、简介，选择只录主臂或从臂同步跟随。同步前核对六轴方向，包含夹爪开合方向。
3. 停止录制后，从下拉列表预览主臂原始录制或从臂实测轨迹。可生成动作草稿，在网页修改每轴起点、终点、运动时间、停留和顺序。
4. 选择「真机试运行此动作」以执行编辑过的关键帧。选择「真机连续回放」以执行同一校准的从臂实测轨迹。两种入口都会连接真实机械臂的控制线程。
5. 选择「组合已保存动作」可编排多个条目。源动作变更或删除后可载入组合复核并重新保存。

仿真预览便于检查动作的大致方向和时序。真机测试要考虑零点、安装方向、线缆、桌面和负载。网页停止属于软件停止，实机测试时应能立即触及物理电源开关。断开从臂或退出服务可能释放扭矩，先托住或放低机械臂。

## 参数、源码和排查

| 参数 | 默认示例与用途 |
| --- | --- |
| `camera.width / height / fps` | `640 / 480 / 30`，控制采集分辨率和目标帧率 |
| `camera.rotation_degrees / flip_horizontal / flip_vertical` | 处理相机安装方向和翻转 |
| `pose.side`、`pose.*confidence` | 跟随侧、姿态与手部置信度 |
| `robot.control_hz` | 示例 `10`，视觉控制基础频率。Agent 和示教有各自的调度节拍 |
| `robot.max_relative_target` | 示例 ID1–ID5 为 `6`、ID6 为 `25`，限制目标与实测位置的差值 |
| `mapping.gains / signs / feature_deadbands` | 控制跟随幅度、方向和死区，ID6 使用专用开合映射 |
| `safety.absolute_limits / relative_limits` | 限制运动工作范围和相对移动幅度 |
| `safety.ema_alpha / max_step_per_cycle` | 示例 `0.55 / 6`，控制平滑和每周期最大变化 |
| `safety.startup_body_abs_limit` | 示例 `200`，连接阶段的启动位置检查。实际动作仍按运动限位 |
| `safety.skip_safe_range_calibration` | 模板默认为 `true`，使用配置范围。设为 `false` 可启用网页安全范围采集 |

核心链路为「浏览器 → Flask → `ControlEngine` → LeRobot → Feetech 舵机」。视觉、模型请求和仿真分别运行，真机写入由同一控制线程管理。

- 改动作和本地口令，查 `agent_schema.py`、`agent_actions.py`。
- 改录制、组合和回放，查 `teach_recording.py`、`teach_trajectory.py`、`simulation.py`。
- 排查连接和运行状态，查 `robot_adapter.py`、`web_engine.py` 与 `logs/control.log`。
- 排查语音延迟，按同一任务查看 `agent plan timing`、`agent speech timing` 与 `agent browser latency`。
- 排查从臂卡顿，对照日志中的目标、实发值、编码器误差和串口周期。

运行以下命令可在无硬件环境检查代码：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -B -m unittest discover -s tests -q
```

65 项现有测试通过，1.2 的本地指令、从臂连续回放与返回连接姿势也已做模拟验证。动作方向、速度、连续回放在实际负载下的效果仍需真机确认。当前视频使用单目图像估计，语音使用浏览器识别及整段合成播放。已知限制、模块关系与开发路线见 [架构与源码指南](ARCHITECTURE.md) 或双击本地 `ARCHITECTURE.html` 离线阅读。

## 致谢与许可

基于戊辰老师的 SO-101 项目思路进行二次开发。机器人控制使用 [LeRobot](https://github.com/huggingface/lerobot)，人手和姿态识别使用 [MediaPipe](https://github.com/google-ai-edge/mediapipe)，六轴仿真使用 [MuJoCo](https://github.com/google-deepmind/mujoco)。

本项目以 [MIT License](LICENSE) 开源。
