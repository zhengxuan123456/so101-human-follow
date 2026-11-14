# SO-101 项目架构与源码指南

版本 `1.2.0` · 2026 年 11 月 14 日

本文介绍 1.2 版的日志、连续回放、本地指令、组合动作和返回连接姿势。部署和入门命令见 [README](README.md)。

## 1. 当前项目能做什么

| 功能 | 怎么使用 | 前置条件与边界 |
| --- | --- | --- |
| 视频跟随 | 电脑摄像头识别右臂与右手，驱动 SO-101 的 ID3–ID6 | 首页连接从臂、连接相机、采集中立姿势、ARM。ID1–ID2 固定。视觉在本地推理 |
| 文字／语音 Agent | 输入或说出「打开夹爪」「点头两次」「摸摸头」，也可以发一句有先后顺序的指令 | 固定明确指令可在本地规划，开放式对话、语义选用自定义动作和部分补充式插话走配置的云模型。真机动作需要从臂连接及安全范围 |
| 返回连接姿势 | 说「返回原位」或「回到连接时的位置」 | 回到本次连接时单独保存的六轴实测位置。工作范围重采集后仍使用这份记录，若超出新范围则提示 |
| 对话和动作并行 | 明确的单动作通过本地校验后可先启动，语音并行准备。多动作和云端动作可等实际开播后开始 | 音频时钟和舵机时钟分别推进。支持中途插话取消旧交互，精确逐句同步和全双工流式对话仍待实现 |
| 对话记忆 | 已完成对话写入本机记忆，按批次生成性格与记忆摘要 | 云端摘要需要已配置的模型和 Key。Agent 名称与音色、语言设置可单独保存 |
| 六轴仿真与动作编辑 | 通过网页滑块、起终点、动作顺序、每轴运动与停留时间编排步骤 | MuJoCo 在 Python 端运行，网页看渲染结果。预览与真机操作各有入口 |
| 主从臂示教 | 填写主臂／从臂串口，使用已有有效校准或现场校准。可只录主臂，也可让从臂实时跟随 | 同步模式会驱动真机，记录主臂样本及从臂编码器反馈。保存动作名称和简介，支持回放与生成编辑草稿 |
| 真机连续回放 | 选择带从臂实测轨迹的录制 →「真机连续回放」 | 从臂先到录制起点，再沿时间轨迹播放。需要同一校准指纹与有效限位。缺少校准记录的老轨迹仍可在 MuJoCo 预览 |
| 动作复用 | 将已保存动作组合，设置每项播放方式、顺序、重复和间隔，可选整体回位 | 组合引用原动作及其版本。引用变动后需要复核，可载入编辑或删除动作 |

**验证状态。** 历史实机记录确认过视频跟随的 ID3 和 ID6。此次 P0–P3 已用模拟舵机验证录制、JSON 保存、起点过渡、连续回放、停止和偏差拦截，组合动作也经过 Flask 接口和 MuJoCo 预览验证。真实动作方向、供电负载下的连贯度、夹爪开合效果和语音起播延迟需要在实际设备上测量。

## 2. 技术栈与运行入口

| 层次 | 技术 | 在项目中的用途 |
| --- | --- | --- |
| 运行环境 | Python 3.10、setuptools | `src` 包布局，Web 与命令行两个入口 |
| 本地服务 | Flask 3.1 系列、Werkzeug、Jinja | 渲染页面、收发 JSON、状态轮询、输出视频与仿真图像 |
| 前端 | 原生 HTML、CSS、JavaScript | 粉色和白色主题，视频、对话、仿真与示教界面，无独立前端构建流程 |
| 相机与视觉 | Windows DirectShow、OpenCV、MediaPipe Tasks、NumPy | 枚举并按名称连接相机，采集、旋转和标注画面，提取人体及手部关键点 |
| 总线 | LeRobot、Feetech 串行舵机 | SO-101 主臂／从臂、编码器、扭矩设置、同步读写、归一化位置和 EEPROM 校准 |
| 动作与仿真 | 本地 Python 轨迹计算、MuJoCo 3.x、MJCF | 构造动作计划、计算时序、渲染 SO-101 模型、回放示教与编排 |
| 语言模型 | OpenAI-compatible Chat Completions 和 Function Calling、httpx | 复杂对话生成结构化语义动作，短指令可由本地解析处理 |
| 识别与合成 | 浏览器 Web Speech、`edge-tts`、浏览器 `speechSynthesis` | 浏览器收集文字识别结果，后端合成在线音色，浏览器播放或回退到系统语音 |
| 数据与日志 | JSON 文件、Python logging、轮转日志 | 动作、示教、配置、偏好、记忆及跟踪耗时。无需数据库、ROS 或 MCP |

- `so101-human-follow-web` → `web_app.main()`。默认地址 `http://127.0.0.1:7860/`，`--http-port` 修改网页端口，`--enable-robot` 允许后续从网页连接硬件。服务启动和机械臂连接是两个操作。 [Web 入口](src/so101_human_follow/web_app.py#L1067)
- `so101-human-follow` → `app.main()`。传统 OpenCV 预览入口，`app.run()` 组织相机、跟随、键盘操作和可选 Rerun 日志视图。 [CLI 入口](src/so101_human_follow/app.py#L108)
- `start_web_preview.bat` 启动预览模式，`start_web_robot.bat` 启动允许真机模式。系统 COM 号可从首页输入，从臂初值来自 `robot.port`，主臂串口在示教面板单独填写。`COM4` 是示例，和网页端口 `7860` 分属不同层。 [启动脚本](start_web_robot.bat) · [配置模板](config.example.json)
- `pyproject.toml` 里的 `[web]` 安装 Flask、edge-tts 与 httpx，`[simulation]` 安装 MuJoCo。视觉模型按下载脚本放到 `models/`，LeRobot 和 OpenCV 使用兼容的本地环境。仿真默认读取相邻目录 `mujoco_sim/scene.xml`，可用 `SO101_MUJOCO_MODEL` 指向自己的 MJCF。本机 `so101-follow` 环境的依赖检查覆盖了 Web、视觉、硬件和仿真。 [包定义](pyproject.toml) · [仿真入口](src/so101_human_follow/simulation.py#L931)

## 3. 整体结构与控制权

```text
本机浏览器
  ├─ app.js ─────────── 状态／连接／相机／校准／ARM ── Flask
  ├─ agent-ui.js ────── 对话／插话／ASR／TTS ────── Flask → AgentService
  ├─ simulation.js ─── 仿真／编排／真机试运行 ─── Flask → ActionLibrary
  └─ teaching.js ───── 主从臂设置／录制 ───────── Flask     │
                                                           ▼
相机线程 → MediaPipe → 特征及预览 ──────────────── ControlEngine
AgentService → 已校验语义计划 → 适配成控制命令 ───────┤
ActionLibrary → 关键帧或从臂轨迹 → 适配成控制命令 ─────┤
TeachRecorder → 主臂读数 ──────────────────────────┤
                                                    ▼
                                     单一串口控制线程 → LeRobot → 六个舵机

网页仿真接口 → SimulationService → MuJoCo 工作线程 → JPEG 与关节状态
```

`create_app()` 装配服务，处理 Cookie、`X-Client-Id`、本机请求和同源写操作检查。`heartbeat()` 发放并续期页面控制权，`submit()` 把命令送入控制队列，`_process_commands()` 在串口线程检查权限和命令代次后执行。`_run()` 汇合校准、示教、视觉跟随和 Agent 的运行节拍。浏览器、LLM 和仿真各自工作，机械臂的真实读写统一由 `ControlEngine` 与驱动适配层处理。 [Flask 装配](src/so101_human_follow/web_app.py#L85) · [控制循环](src/so101_human_follow/web_engine.py#L3706)

视频 ARM、Agent／编辑器真机动作、主从同步示教和手动校准共享物理控制权。`DISARM`／停止撤销活动计划并让后续旧命令失效，取消中的动作可在控制线程读编码器并保持实测姿态。断开连接按配置释放扭矩，机械臂可能下坠。网页停止属于软件控制，物理断电可独立于网页运行状态使用。 [控制权及停止](src/so101_human_follow/web_engine.py#L308) · [编码器保持](src/so101_human_follow/web_engine.py#L1953)

## 4. 模块与关键方法索引

| 路径 | 关键方法／对象 | 输入、输出与下游 |
| --- | --- | --- |
| [web_app.py](src/so101_human_follow/web_app.py#L85) | `create_app`、`dispatch_agent_plan`、`preempt_interaction`、`main` | HTTP 与本地服务的适配。将语义动作、动作库引用转换成控制命令，返回任务状态和接口错误 |
| [web_engine.py](src/so101_human_follow/web_engine.py#L118) | `heartbeat`、`submit`、`disarm`、`_run`、`_vision_loop` | 管理租约、硬件与相机状态。控制线程接收命令，视觉线程采集和识别，状态供网页轮询 |
| [config.py](src/so101_human_follow/config.py#L121) | `load_config`、`validate_config`、`AppConfig`、`ALL_JOINTS` | 将 JSON 转为类型化的相机、映射、机器人、安全配置；检查关节集合、数值与方向 |
| [camera_device.py](src/so101_human_follow/camera_device.py#L127) | `list_camera_names`、`camera_index_by_name` | Windows DirectShow 枚举设备名称，返回 OpenCV 可使用的设备索引 |
| [pose_detector.py](src/so101_human_follow/pose_detector.py#L91) | `MediaPipePoseDetector.detect`、`_select_hand`、`orient_camera_frame`、`annotate_frame` | 从画面生成 `PoseDetection`，关联手部与目标腕部，并绘制预览 |
| [pose_math.py](src/so101_human_follow/pose_math.py#L185) | `extract_arm_feature_channels`、`extract_hand_feature_channels`、`median_features` | 从关键点提取各通道特征及可靠性，供视觉跟随与中立姿势采集 |
| [safety.py](src/so101_human_follow/safety.py#L134) | `MotionController.compute`、`reset_to`、`signed_deadband`、`ema`、`slew_limit` | 把人体特征映射为六轴归一化目标，应用固定关节、死区、平滑、限位、限速 |
| [safe_range.py](src/so101_human_follow/safe_range.py#L21) | `SafeRangeProfile`、`load_safe_range_profile`、`save_safe_range_profile` | 管理固定位置、视觉工作范围、机器人 ID 与校准指纹 |
| [robot_adapter.py](src/so101_human_follow/robot_adapter.py#L31) | `SO101RobotAdapter.connect`、`read_positions`、`send_agent_targets`、`hold_current`、`SO101CalibrationSession` | 对接 Feetech 和 LeRobot。返回实测、实发位置，处理校准核对、同步写入和扭矩 |
| [speech_normalizer.py](src/so101_human_follow/speech_normalizer.py#L81) | `normalize_speech` | 转写标准化、名称前缀、常见误识别和次数修正，以及本地 STOP 辨认 |
| [agent_schema.py](src/so101_human_follow/agent_schema.py#L176) | `build_fixed_intent_plan`、`build_system_prompt`、`InteractionPlan` | 完整固定指令本地解析；定义云端工具格式、动作名称和参数边界 |
| [agent_client.py](src/so101_human_follow/agent_client.py#L82) | `OpenAICompatibleClient`、`validate_base_url` | 把请求发送给兼容模型，验证工具调用、处理超时、取消和错误脱敏 |
| [agent_service.py](src/so101_human_follow/agent_service.py#L412) | `AgentService.submit_text`、`_process_task`、`_request_cloud_plan`、`start_ready` | 管理对话任务、插话代次、到音频起播后的派发、动作回调与记忆线程 |
| [agent_memory.py](src/so101_human_follow/agent_memory.py#L53) | 记忆数据校验、存取与对话批次管理 | 保存 `persona`、`user_facts`、`recent_summary` 和近期消息，为后续云端对话提供上下文 |
| [speech_synthesis.py](src/so101_human_follow/speech_synthesis.py#L70) | `SpeechSynthesizer` | 在线音色合成、缓存、取消、超时与体积限制，结果交给浏览器 |
| [agent_actions.py](src/so101_human_follow/agent_actions.py#L742) | `build_agent_motion_plan`、`agent_frame_schedule`、`calculate_agent_motion_timing`、`step_toward_agent_target` | 将允许的动作展开为 `AgentKeyframe`，为并行与顺序运动计算时间和最大每周期变化 |
| [motion_profile.py](src/so101_human_follow/motion_profile.py#L10) | `cosine_progress` | 计算关键帧内的缓入缓出进度，供仿真和 Agent 控制使用 |
| [teach_recording.py](src/so101_human_follow/teach_recording.py#L51) | `TeachRecorder.start_recording`、`poll`、`record_follower_sample`、`stop_recording` | 连接主臂、同步采样、保存主从轨迹与范围、写入录制时从臂校准标识 |
| [teach_trajectory.py](src/so101_human_follow/teach_trajectory.py#L21) | `FollowerTrajectory`、`recording_position`、`editable_steps` | 连续轨迹校验、时间采样、每段限速及关键帧提取 |
| [simulation.py](src/so101_human_follow/simulation.py#L203) | `ActionLibrary`、`expand_action_frames`、`SimulationService` | 动作与轨迹文件管理、组合展开、MuJoCo 模型与离屏预览 |
| [app.py](src/so101_human_follow/app.py#L108)、[viewer.py](src/so101_human_follow/viewer.py#L8) | `run`、`RerunViewer.log` | 命令行流程和可选诊断视图 |

| 前端文件 | 负责的内容 |
| --- | --- |
| [templates/index.html](src/so101_human_follow/templates/index.html) | 首页连接、相机与对话弹窗、设置、动作编辑和示教的 HTML 骨架 |
| [static/app.js](src/so101_human_follow/static/app.js) | 控制标识、心跳与状态、连接／断开、视频设备、校准、倒计时和 STOP |
| [static/agent-ui.js](src/so101_human_follow/static/agent-ui.js#L705) | `sendMessage`、`startRecognition`，语音合成请求、播放、语速选择、插话、任务状态与中英语音选项 |
| [static/simulation.js](src/so101_human_follow/static/simulation.js#L1054) | 滑块与顺序、关键帧／组合编辑、列表、仿真时间、`executeTrial` 和实测轨迹按钮 |
| [static/teaching.js](src/so101_human_follow/static/teaching.js) | 主从臂串口、校准、录制模式、方向确认、名称简介、范围和草稿生成 |
| [style.css](src/so101_human_follow/static/style.css)、[agent-theme.css](src/so101_human_follow/static/agent-theme.css)、[simulation.css](src/so101_human_follow/static/simulation.css) | 基础样式、粉白主题、仿真和动作列表布局 |

## 5. 视觉跟随、校准与六舵机映射

### 三种数据各自解决的问题

| 数据 | 何时采集 | 编码及影响 |
| --- | --- | --- |
| 舵机基础校准 | 首次设置、更换硬件或从臂校准文件与 EEPROM 不一致时 | 原始编码器值和零点偏移。网页流程释放扭矩、确认六轴中位、收集 90 秒范围并保存。ID5 单圈基础校准上限为 4095，保存时核验写回结果 |
| 机械臂工作范围 | 连接后按配置使用已有范围、自动应用配置范围，或进入人工范围采集 | LeRobot 归一化位置。人工流程给 ID1–ID2 约 10 秒确认固定姿态，采集 ID3–ID6 约 60 秒，再回到工作起点保存。档案绑定机器人 ID 和校准指纹 |
| 人体中立姿势 | 从臂和相机就绪后，视频跟随 ARM 前 | 人体特征统计。前端有约 3 秒准备时间，后端收集约 30 秒。各通道可在不同帧有效，结束时检查四个通道都有样本。Agent 对话动作和示教各按自己的前置检查运行 |

`_connect_robot()` 通过 `SO101RobotAdapter.connect()` 连接并核对校准；`_save_base_calibration()` 调用 `SO101CalibrationSession.finish()` 处理 EEPROM 和 LeRobot 文件；`_finish_safe_range()` 保存工作范围；`_start_neutral_collection()`、`_update_tracking()` 和 `_arm()` 管理人体中立值和视觉启动。公开模板启用了 `skip_safe_range_calibration`，使用配置的工作范围和连接时的固定姿态，可跳过单独的工作范围采集。 [从臂连接](src/so101_human_follow/web_engine.py#L2308) · [基础校准](src/so101_human_follow/robot_adapter.py#L459) · [视觉 ARM](src/so101_human_follow/web_engine.py#L3020)

### 位置值与方向

| ID | LeRobot 关节 | 视觉输入 | 配置模板的绝对位置范围 |
| --- | --- | --- | --- |
| 1 | `shoulder_pan.pos` | 固定，底座水平旋转 | `-70..70` |
| 2 | `shoulder_lift.pos` | 固定，肩部抬落 | `-70..70` |
| 3 | `elbow_flex.pos` | 右侧大臂与小臂夹角 | `-70..70` |
| 4 | `wrist_flex.pos` | 右侧上臂抬升／降低 | `-100..100` |
| 5 | `wrist_roll.pos` | 手掌旋转 | `-100..100` |
| 6 | `gripper.pos` | 握拳／张手 | `0..100` |

表格使用 LeRobot 归一化位置。EEPROM 原值、LeRobot 归一化值、MuJoCo 模型弧度和真实物理角度是不同量。启动位置检查 `startup_body_abs_limit` 与运行目标范围也有各自用途；`±200` 配置数值的单位仍是归一化位置。Agent、手工编排和示教允许六轴参与，由 `agent_joint_limits()` 将配置绝对范围与已知工作范围取交集。 [关节与配置](src/so101_human_follow/config.py) · [六轴工作范围](src/so101_human_follow/web_engine.py#L1365)

### 从相机到串口

1. `connect_camera()` 按设备名称打开 Windows 相机，视觉线程 `_vision_loop()` 取图、应用旋转或翻转，再调用 `MediaPipePoseDetector.detect()`，将带标注画面通过 `/video_feed` 发送。断开相机释放设备并清除视觉会话的中立数据。
2. `extract_arm_feature_channels()` 和 `extract_hand_feature_channels()` 分开计算肘、抬臂、手掌旋转和开合，关联腕部并记录有效通道。画面范围能覆盖需要的右肩、肘、腕或手部即可采集相应通道。
3. `MotionController.compute()` 按「当前特征减中立值 → 死区 → `gain × sign` → 限位 → EMA → 单次变化上限」产生目标，某通道暂失时保持上一次目标。ID6 单独用 `clamp((hand_open - 0.35) / 0.20, 0, 1)` 将开合压到可用行程，再应用方向、工作范围与速度约束。
4. `_send_if_due()` 以配置的跟随控制节拍写从臂。多人、目标切换、关键点整体超时和相机中断可解除跟随。视觉接近用人体／手部在画面中的相对尺寸估计，`TOO_CLOSE` 可停止视觉跟随。`SLOW`／`UNKNOWN` 当前是报告状态，系统没有基于它们实现精确距离闭环。 [检测器](src/so101_human_follow/pose_detector.py#L91) · [运动映射](src/so101_human_follow/safety.py#L134) · [视觉保护](src/so101_human_follow/web_engine.py#L3092)

## 6. Agent、语音、记忆与插话

```text
网页文字或电脑麦克风 → 浏览器生成文字 → sendMessage()
  → POST /api/agent/message → normalize_speech()
    ├─ 明确停止 ─────────────→ 本地取消、实测位置保持
    ├─ 完整固定指令 ─────────→ build_fixed_intent_plan()
    └─ 开放对话／待处理上下文 → OpenAICompatibleClient → execute_interaction_plan
  → InteractionPlan → dispatch_agent_plan() → ControlEngine.submit()
  → 浏览器播放答复                     └─ 关键帧／连续轨迹 → 编码器反馈
```

- **本地分支。** `build_fixed_intent_plan()` 识别完整的已知命令或最多 12 项的明确序列。否定式、疑问或未识别完整的输入走后续判断。独立「你好」默认 ID6 开合两次，并用配置的 Agent 名称自我介绍；「摸摸头」由本地动作表生成向前低头意图，停留约 2 秒后回位；独立抬头是另一动作。三至五轴组合、点头和挥手等由本地实现决定参与轴、位移与回位，实际前后方向以机械臂校准和试跑验证。 [固定指令](src/so101_human_follow/agent_schema.py#L234) · [动作展开](src/so101_human_follow/agent_actions.py#L742)
- **云端分支。** 模型调用的工具名为 `execute_interaction_plan`。本地检验 `version`、`reply`、`actions` 和 `requires_confirmation`，再逐个验证动作参数。自定义动作目录向模型提供 `id`、`name`、`description`，模型用 `perform_custom_action` 选择 ID。实际轨迹由本地动作库读取和重新校验。 [工具规范](src/so101_human_follow/agent_schema.py#L352) · [计划派发](src/so101_human_follow/web_app.py#L104)
- **音频时序。** 本地明确单动作可在准备语音时派发。需要同步起步的计划可先进入 `READY`，前端等实际开播后调用 `/api/agent/start`，通过 `/api/agent/playback` 报告播放、暂停和结束。暂停时控制线程可保持实测位置。语速改变音频播放速度，动作时长由舵机限位、限速、反馈和动作编排决定。内部任务 `COMPLETED` 代表已派发，真机完成看机械臂动作状态 `COMPLETE`。 [Agent 任务](src/so101_human_follow/agent_service.py#L791) · [播放状态](src/so101_human_follow/web_app.py#L887)
- **插话与顺序。** 浏览器用序号和 `AbortController` 淘汰旧请求。`preempt_interaction()` 把已完成、未完成与当前动作整理成快照，取消旧任务并请求读取实测位置。补充式插话可把上一轮意图与本轮内容交给云端理解，已完成的部分可从待办中扣除。被打断的连续录制回放不会自动从旧位置续播。文字发送和语音中间结果都能触发打断；浏览器识别是否能持续监听取决于其实现及回声环境。 [插话入口](src/so101_human_follow/web_app.py#L329) · [动作快照](src/so101_human_follow/web_engine.py#L1394)
- **在线音色和存储。** `SpeechSynthesizer` 整段生成 MP3 后交给浏览器，后端有内存缓存和超时，失败可尝试浏览器系统语音。浏览器录音识别与在线 TTS 各自可能依赖外部服务。Agent 偏好存名称、服务地址、模型、音色和语言，API Key 只保存在进程内存。 [语音实现](src/so101_human_follow/speech_synthesis.py#L70) · [Agent 偏好](src/so101_human_follow/agent_service.py#L151)
- **记忆。** 对话完成后异步写 `agent_memory.json`，每批 20 条消息约相当于 10 轮用户和助手对话。云端摘要提取性格、用户事实和近期摘要。云端后续对话可读摘要及少量近期消息，纯本地固定指令规划可直接执行。记忆作为文本上下文使用，机械动作权限仍由代码校验。 [记忆存取](src/so101_human_follow/agent_memory.py#L53) · [后台摘要](src/so101_human_follow/agent_service.py#L730)

## 7. 示教、动作文件与 MuJoCo

### 主从臂录制与连续回放

1. `TeachRecorder.start_recording()` 检查主臂六轴校准、读取设备校准并关闭主臂扭矩，主臂可手动摆动。同步模式还检查从臂连接、有效工作范围和六轴方向。夹爪的反向映射用 `100 - value`，其他轴用 `-value`。
2. `TeachRecorder.poll()` 按目标约 30 Hz 获取主臂数据；`_send_teach_if_due()` 使用最新目标，从臂发送节拍至少按 25 Hz 目标调度，具体达成频率取决于串口负载。通过时间相关平滑和步长限制跟随，主臂样本超过约 150 毫秒时暂停发送。`record_follower_sample()` 记录编码器读数和采样时间。 [主臂处理](src/so101_human_follow/teach_recording.py#L221) · [从臂同步](src/so101_human_follow/web_engine.py#L1305)
3. 录制停止后 `ActionLibrary.save_recording()` 保存到 `action_library/<id>.json`。每路最多 3600 帧，模板代码上限 119 秒，含录制文件最大 4 MiB。录制失败时可保留可读轨迹供预览，串口故障或保持失败会取消该次录制的真机回放资格。 [存储](src/so101_human_follow/simulation.py#L441) · [轨迹失效处理](src/so101_human_follow/teach_recording.py#L340)
4. 「回放原始录制」可查看主臂，「预览从臂轨迹」查看编码器记录。「生成动作草稿」优先用从臂样本，由 `editable_steps()` 提取关键姿势和停留，把自定义步骤与原录制存进同一个 JSON。 [草稿生成](src/so101_human_follow/simulation.py#L400)
5. 「真机连续回放」向 `/api/actions/<id>/execute` 提交 `playback: follower_trajectory`。`execution_blocks()` 校验录制指纹、限位和动作版本，`_start_agent_plan()` 从实测姿态开始，先到录制首帧，再沿时间连续下发。`_send_agent_if_due()` 用上一目标和编码器反馈比较持续偏差，当前检测窗口约 0.35 秒。连续回放结尾检查舵机到位，支持软件停止。 [回放入口](src/so101_human_follow/web_app.py#L805) · [执行校验](src/so101_human_follow/web_engine.py#L1496)

### 三种保存形式

| 形式 | JSON 关键字段 | 可调用路径 |
| --- | --- | --- |
| 原始录制 | `recording.samples`、`follower_samples`、`follower_calibration_fingerprint`、`calibration_ranges` | 仿真原始回放、生成草稿；合格的从臂实测轨迹可经独立入口真机连续回放 |
| 编辑动作 | `name`、`description`、`frames`、`return_to_start`、`limits_fingerprint` | 仿真预览、真机试运行、模型按名称及简介选择 |
| 组合动作 | `kind: composition`、`items`、每项 `action_id/playback/repeat/hold_s/revision`、整体 `return_to_start` | 合并已保存动作，可把关节步骤和从臂实测轨迹放在同一组合中 |

顶层 `schema_version` 目前为 1，各轴位置键采用完整 `.pos` 名称。新增录制的校准指纹来自从臂六轴的 ID、方向、零点与范围。组合以源动作内容计算引用版本，修改轨迹或删除源动作后会提示复核。普通动作支持 1–32 个编辑姿势；组合最多 12 项、每项最多重复 8 次、名义总时长上限 120 秒，展开后最多 512 步。轨迹运动限速后实际时间可能更长。 [动作库验证](src/so101_human_follow/simulation.py#L225) · [组合展开](src/so101_human_follow/simulation.py#L337)

### 运动公式与仿真区别

| 路径 | 进度计算及用途 |
| --- | --- |
| 视觉跟随 | 连续特征 → 目标 → EMA 和单周期步长，不构造整段离线动作 |
| 同步示教 | `α = 1 − exp(−Δt/0.04)` 平滑最新主臂目标，保留原本的配置速度上限，防止慢串口一次积累过大步长 |
| 关键帧 | `u = clamp(t/T, 0, 1)`，`s(u) = (1 − cos(πu))/2`，`q = q₀ + Δq × s(u)`。起终点速度为零，峰值约为 `π|Δq|/(2T)`。每个关键帧可有独立运动及停留时间，真机需依据到位反馈继续 |
| 从臂连续轨迹 | 时间戳之间做线性插值。`FollowerTrajectory.limited_to()` 对超速段延长 `Δt` 到至少 `max(各轴位移 / 各轴有效速度)`，其余段保留原时长。控制循环仍受单次变化上限、采样抖动和实测偏差检查约束 |

仿真 `SimulationService._open()` 加载 MJCF 与六个关节索引，`_to_radians()` 按模型范围换算；`_advance()` 分流原轨迹与编排动作。原轨迹设置模型姿态后运行 `mj_forward()`，侧重复现关节路径与节奏。编排仿真会推进模型物理步，离屏渲染成 JPEG。接触点提示可用于排查模拟模型，真实桌面、线缆、负载和零点还需实物验证。 [轨迹实现](src/so101_human_follow/teach_trajectory.py#L21) · [余弦插值](src/so101_human_follow/motion_profile.py#L10) · [仿真推进](src/so101_human_follow/simulation.py#L757)

## 8. 从需求定位到代码

| 想修改什么 | 从这里进代码 | 修改后看什么 |
| --- | --- | --- |
| 修改某个动作姿态或增加一个内置动作 | `agent_actions.py` 里的 `move_joint`、六轴封装 `move_id1` 到 `move_id6`、`parallel`、`sequence`、`build_agent_motion_plan`。若要增加自然语言触发，检查 `agent_schema.py` 和 `web_app.py` 的语义映射 | 参与轴、实际方向、运动起点、停顿、回位与 STOP。可以先做成动作库条目，从而用网页编辑和预览 |
| 单独调大夹爪视觉开合或改正方向 | `pose_math.py` 的手部特征 → `safety.py` 的 ID6 开合映射 → `mapping.signs`、`absolute_limits` 和 `robot.max_relative_target` | 页面读数是否从握拳端切到张手端、实发目标是否被位置差限制、编码器能否跟上 |
| 修复主从臂顿挫 | `web_engine.py` 的 `_poll_teach`、`_send_teach_positions` 与 `robot_adapter.py` 的 `send_agent_targets` | `teach motion telemetry` 里的 `leader_read_ms`、`leader_sample_age_ms`、串口 `io_ms`、有效写频率和误差 |
| 优化多个动作衔接 | `agent_frame_schedule()` → `_send_agent_if_due()`；录制路径从 `FollowerTrajectory.position_at()` 查起 | 到达关键帧、停留与回位的时序，区别设计停顿、编码器等待、轨迹限速与串口负载 |
| 减少语音等待或改善插话 | `static/agent-ui.js` 的识别、发送和播放 → `/api/agent/message`、`/api/agent/interrupt`、`/api/agent/speech` → `AgentService` | 同一任务的识别完成、规划、首声音和动作启动。检查旧的回复、音频或动作是否被二次派发 |
| 调整动作被 Agent 命中的方式 | 先改动作 JSON 中 `name`／`description`，再看 `web_app.py` 动作清单与 `agent_schema.py` 工具参数 | 用户语义与列表中动作是否唯一匹配，只有有效 ID 能进入本地执行 |
| 仿真角度与真机偏差 | `simulation.py` 的 `_to_radians()`、`_to_normalized()`，配合 `mujoco_sim/scene.xml` 与从臂校准值 | 逐轴比对零位、正负方向、角度比例和夹爪端点，再对组合姿态 |

### 接口与数据示例

| 目的 | 接口 |
| --- | --- |
| 网页控制权与状态 | `GET /api/status`，`GET /api/joints` |
| 相机 | `GET /api/camera/devices`，`POST /api/camera/connect`，`POST /api/camera/disconnect`，`GET /video_feed` |
| 从臂和跟随 | `POST /api/connect`，`/api/disconnect`，`/api/base-calibration/*`，`/api/safe-range/*`，`/api/neutral`，`/api/arm`，`/api/stop` |
| 示教采集 | `GET /api/teach/status`，`POST /api/teach/calibration/*`，`POST /api/teach/recording/start` 与 `.../stop` |
| 动作库与仿真 | `GET/POST /api/simulation/actions`，`GET /api/simulation/recordings`，`POST /api/simulation/recording/draft`、`/api/simulation/play`、`/api/simulation/play_recording` |
| 真机动作 | `POST /api/actions/<id>/execute`，`POST /api/joints/move` |
| 对话与音频 | `POST /api/agent/message`、`/interrupt`、`/start`、`/playback`、`/speech`、`/latency`、`/confirm`、`/cancel`；子路径均在 `/api/agent` 下 |

例如「点头两次」对应的语义计划可以表示为下面的 JSON。上层只选择动作意图，硬件执行仍使用当前工作范围和实测位置。

```json
{
  "version": "1.0",
  "reply": "好呀。",
  "actions": [
    {"name": "perform_gesture", "arguments": {
      "gesture": "nod", "count": 2, "intensity": "normal", "speed": "normal"
    }}
  ],
  "requires_confirmation": false
}
```

一项组合动作的编辑参数如下。`action_id` 填动作库中实际 ID；`revision` 由保存过程绑定。示例只展示单项配置。

```json
{
  "action_id": "0123456789abcdef0123456789abcdef",
  "playback": "follower_trajectory",
  "repeat": 2,
  "hold_s": 0.4
}
```

## 9. 工程文件、测试与故障排查

| 文件或目录 | 用途 |
| --- | --- |
| `config.example.json`、`config.json` | 分别是可公开模板和本机硬件／相机／限位配置 |
| `safety_profile.json`、LeRobot 校准目录 | 前者保存采集的工作范围，后者保存底层舵机校准。校准重新写入后应复核旧动作和录制的兼容性 |
| `action_library/` | 命名动作、介绍、编排、原始轨迹、备份及删除标记 |
| `agent_preferences.json`、`agent_preferences.backup.json` | 可恢复的非密钥设置 |
| `agent_memory.json` | 对话记忆、近期消息及摘要进度 |
| `logs/control.log` | 滚动保存控制日志。单份上限约 5 MiB，保留最多 5 份归档 |
| `models/`、`scripts/` | MediaPipe `.task` 和下载脚本。MuJoCo MJCF 及网格由独立的 `mujoco_sim` 目录提供 |
| `tests/` | 位姿数学、限位、校准、驱动适配、配置、Flask 路由和控制锁测试 |

本机数据的忽略项见 [.gitignore](.gitignore)。本机配置模板可用于团队分发；含有实际轨迹、日志、音频相关信息与个人对话的文件应按项目数据范围处理。

| 现象 | 先看哪一层 | 如何判断 |
| --- | --- | --- |
| 连接从臂失败 | `robot_adapter.py`／`control.log` | 区分 COM 口或舵机无反馈、EEPROM 校准不匹配、归一化启动位置越界。`Missing motor IDs` 和校准哈希不一致对应不同阶段 |
| 相机黑屏或选错 | `/api/camera/devices`、相机状态 | 核对设备名称、权限和占用。相机断开后再连接指定设备，确认预览有新帧 |
| 录制或真机试跑只到起点／卡顿 | `agent frame timeline`、`agent keyframe measured arrival`、`teach motion telemetry`、`follower_replay motion telemetry` | 确认使用的是关键帧试运行还是连续回放；对比 `target`、`planned`、`sent`、`measured`、P50/P95 周期和各阶段耗时 |
| 对话语音迟、插话慢 | `agent plan timing`、`agent speech timing`、`agent browser latency` | 同一任务 ID 对齐排队、模型规划、合成、音频下载、首次播放和动作确认，找出等待最长的一段 |
| 模型选错自定义动作或动作不可用 | 动作名、简介、库列表状态、回放接口错误 | 检查动作名称是否冲突、简介是否具体、引用是否修改或删除、当前校准指纹是否匹配 |

在项目目录用已激活的 Python 环境运行现有测试。

```powershell
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -q
```

1.2 发布检查通过 65 项单元测试。测试使用公开配置，覆盖对话动作可在视觉不可用时运行，以及视觉跟随在判断人或手太近时会停止。在 `so101-follow` 环境中加载本地 MJCF，获得了六轴状态和 JPEG 渲染结果。实际回位、负载与连贯度仍需按实物验证。 [测试源码](tests/test_web_api.py)

## 10. 下一步开发优先级与参考项目

| 优先级 | 要做的事 | 改动位置与验收方式 |
| --- | --- | --- |
| 先做 · 真机基线 | 对照连续回放和旧关键帧试运行，分别测短动作、不同起点、回位和 STOP。记录主臂、从臂报文、编码器与音频的同段日志 | 沿用 `ControlEngine`。用日志区分主臂样本慢、串口节拍波动、舵机跟不上或关键帧设计停顿。复现「只到起点」时记录最后一帧状态和错误字段 |
| 再做 · 轨迹衔接 | 将采样轨迹的时标处理整理为共享函数，给仿真和实机提供同一限速后目标曲线。对跨段连接处理连续速度，保留显式停顿 | 从 `teach_trajectory.py`、`motion_profile.py`、`SimulationService` 和 `_send_agent_if_due()` 下手。用同条轨迹对比原始时长、限速后时长、画面和真机编码器；停止后不得继续发旧目标。参考 lampOS 录制时标与重采样，以及 LeLamp Runtime 的切换和回位 |
| 并行 · 语音响应 | 先用 P0/P2 日志确定瓶颈；将首句与后续多句分段，评估流式 TTS 与语音活动检测，区分环境回声和真实插话 | 复用 `AgentService` 的请求代次和 `/api/agent/interrupt`。分别统计文字发送到首动作、语音结束到首声、插话到旧声停止；同时回归动作 STOP |
| 随后 · 动作库维护 | 给动作增加可检索的触发示例或标签，显示引用变化、校准版本、超限轴和预估时长，完善编辑与回放诊断 | 复用已有 `name`、`description`、`revision` 和 JSON 校验。验收典型同义词调用、重名时澄清、被删动作提示复核。参考 YareLampGo 的组合技能和无硬件验证思路 |
| 随后 · 仿真对应与保护 | 单轴多点比对真机和模型，通过日志定位方向和偏移，再验证机身、桌面、夹爪及线缆间的空间 | 增加逐轴映射标定表与安全回归动作。用真实结构复核仿真接触结果。项目当前重点是夹爪与机械臂动作，注视和灯光留待以后评估 |

- **lampOS** 的运动播放文档说明如何按原始时间戳重采样、延长超速段，并在模拟设备中复用时间规则。可借鉴对时和可测量的验收方法，SO-101 的单位、夹爪和安全阈值应按本项目重新标定。 [lampOS Motion Playback](https://github.com/autonomous-ai/lampOS/blob/main/robots/lamp/docs/motion-playback.md) · [语音事件与实时接口](https://github.com/autonomous-ai/lampOS/blob/main/docs/mqtt.md)
- **LeLamp Runtime** 的 `smooth_animation.py` 提供有名录制、过渡、连续播放和回待机的参考。适合对照我们的「从实测位置切入 → 播放 → 按需回位」。其台灯关节定义与项目的六轴 SO-101 有差异。 [LeLamp Runtime 源码](https://github.com/humancomputerlab/lelamp_runtime/blob/main/smooth_animation.py)
- **YareLampGo** 展示了本地控制、命名技能组合与无硬件使用方式。适合参考本地固定指令和动作组合流程。实际复用源码前应按被复用文件确认其许可，项目页面标注 GPL-3.0，本项目使用 MIT。 [YareLampGo 仓库与许可](https://github.com/ai-maker-lab/YareLampGo) · [本项目许可](LICENSE)
- **LeRobot、MediaPipe 与 MuJoCo** 是排查基础行为的首选。硬件校准看 LeRobot，视觉特征和手部分类看 Google MediaPipe，渲染、接触与模型字段看 MuJoCo。 [LeRobot SO-101](https://huggingface.co/docs/lerobot/main/en/so101) · [Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) · [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) · [MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html) · [SO-101 MJCF 参考](https://huggingface.co/lerobot/so101-tabletop-mujoco)
