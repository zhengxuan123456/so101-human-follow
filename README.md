# SO-101 Human Follow

使用普通电脑摄像头，让 LeRobot SO-101 从臂实时跟随右臂和右手动作的实验项目。

项目通过 MediaPipe 提取右肩、右肘、右腕及右手关键点，计算肘部夹角、上臂姿态、手掌旋转和张合状态，再映射为 SO-101 舵机位置指令。全部视觉推理在本机完成，不依赖大语言模型、云端推理或 ROS。

> 当前是可运行的实验版本：ID3 肘部跟随和 ID6 夹爪张合已经在真机上跑通。不同机械臂、摄像头位置和使用者需要继续调整映射比例、方向与响应速度。

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
├─ config.example.json          # 可公开的配置模板
├─ models/                      # 视觉模型（不提交 Git）
├─ scripts/                     # 模型下载脚本
├─ src/so101_human_follow/
│  ├─ pose_detector.py          # Pose/Hand 推理
│  ├─ pose_math.py              # 关键点转控制特征
│  ├─ safety.py                 # 映射、平滑、限位与限速
│  ├─ robot_adapter.py          # LeRobot/串口适配
│  ├─ web_engine.py             # 控制状态机
│  └─ web_app.py                # Flask 服务
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
Copy-Item config.example.json config.json
```

修改 `config.json` 中的：

- `robot.port`：机械臂实际串口，例如 `COM4`。
- `robot.id`：必须与 LeRobot 校准时使用的机器人 ID 一致。
- `camera.index`：电脑摄像头通常为 `0`，请先通过预览确认。
- `pose.side`：默认跟随右臂，值为 `right`。

本机 `config.json`、运行日志、摄像头画面、机械臂校准和安全范围均被 `.gitignore` 排除。

## 运行

### 1. 纯视觉预览

双击 `start_web_preview.bat`，或运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m so101_human_follow.web_app --config config.json --http-port 7860
```

打开 <http://127.0.0.1:7860/>。这一模式不会连接机械臂，也不会发送舵机指令。

请确认画面中能完整看到右肩、右肘、右腕和右手，并观察关键点是否稳定。

### 2. 允许网页连接真机

确认预览正常后，关闭预览服务并双击 `start_web_robot.bat`，或运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m so101_human_follow.web_app --config config.json --http-port 7860 --enable-robot
```

推荐操作顺序：

1. 清空机械臂活动区域，并让实体电源开关保持触手可及。
2. 在网页中连接从臂；连接成功后仍为 `DISARMED`。
3. 采集中立姿势，等待进度完成。
4. 确认识别稳定后点击 ARM。
5. 先做一次很小的动作，确认各舵机方向正确。
6. 结束时先停止跟随，再托住机械臂并断开/释放扭矩。

基础舵机校准只在 LeRobot 校准不匹配时使用，不需要每次启动都重新校准。

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
| `neutral_sample_count` | `30` | 中立姿势每个通道需要的有效帧数 |

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

## 测试

不连接摄像头和机械臂即可运行：

```powershell
python -m unittest discover -s tests -v
```

当前版本共有 57 项单元测试。

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
