# 视觉组大作业：路线 A

本仓库完成路线 A：装甲板检测与位姿估计。

程序逐帧读取比赛录像，使用提供的 ONNX 模型检测装甲板，恢复四个角点到原始图像坐标，使用 NMS 抑制重复检测，并通过平面 PnP 计算装甲板四个物理角点在相机坐标系中的三维坐标。结果以 JSON Lines 格式输出。

## 环境要求

- Python 3.10 或更高版本
- CPU 即可，不需要 CUDA

创建虚拟环境：

```bash
python -m venv .venv
```

Windows Git Bash：

```bash
./.venv/Scripts/python.exe -m pip install -r detection/requirements.txt
```

Linux / macOS：

```bash
./.venv/bin/python -m pip install -r detection/requirements.txt
```

## 运行方法

处理小装甲板视频：

```bash
./.venv/Scripts/python.exe detection/detection.py \
  --model detection/assets/0526.onnx \
  --video detection/assets/small.mp4 \
  --calibration detection/assets/calibration.yaml \
  --output detection/small.jsonl \
  --confidence 0.5 \
  --nms-iou 0.5
```

处理大装甲板视频：

```bash
./.venv/Scripts/python.exe detection/detection.py \
  --model detection/assets/0526.onnx \
  --video detection/assets/large.mp4 \
  --calibration detection/assets/calibration.yaml \
  --output detection/large.jsonl \
  --confidence 0.5 \
  --nms-iou 0.5
```

调试时可用 `--max-frames` 限制处理帧数：

```bash
./.venv/Scripts/python.exe detection/detection.py \
  --model detection/assets/0526.onnx \
  --video detection/assets/small.mp4 \
  --calibration detection/assets/calibration.yaml \
  --output detection/test.jsonl \
  --confidence 0.5 \
  --nms-iou 0.5 \
  --max-frames 10
```

Linux / macOS 请将上述命令中的 Python 路径替换为 `./.venv/bin/python`。

## 实现流程

1. 使用 OpenCV 按顺序读取视频帧。
2. 通过居中 letterbox 将画面等比例缩放到 640 × 640，补边值为 114。
3. 将图像转换为 RGB、NCHW、float16，并归一化到 `[0, 1]`。
4. 使用 ONNX Runtime CPU 后端运行模型。
5. 对 objectness logit 使用 sigmoid，并按照置信度阈值筛选候选。
6. 解码颜色、图案以及左上、左下、右下、右上四个角点。
7. 使用缩放比例和补边量将角点恢复到原始图像坐标。
8. 使用四角点的轴对齐包围框执行类别无关 NMS。
9. 根据图案类别选择大小装甲板物理模型。
10. 使用 `SOLVEPNP_IPPE` 产生平面位姿候选，并通过 LM 优化、正深度检查和重投影误差选择有效解。
11. 将旋转和平移同时作用于四个物理角点，得到相机坐标系中的三维坐标。
12. 每帧输出一个完整 JSON 对象。

## 装甲板尺寸

| 类型 | 宽度 | 高度 |
| --- | ---: | ---: |
| 小装甲板 | 0.135 m | 0.056 m |
| 大装甲板 | 0.230 m | 0.056 m |

图案 `1` 和 `base_big` 使用大装甲板模型，其余图案使用小装甲板模型。

## 输出格式

输出采用 UTF-8 JSON Lines，每行对应一帧。没有检测的帧示例：

```json
{"frame_index": 0, "timestamp_s": 0.0, "detections": []}
```

每个检测包含：

- `color`
- `pattern`
- `confidence`
- `corners_px`
- `corners_camera_m`
- `reprojection_error_px`

PnP 失败时仍保留检测，并将 `corners_camera_m` 和 `reprojection_error_px` 写为 `null`。

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | ---: |
| `--model` | ONNX 模型路径 | 必填 |
| `--video` | 输入视频路径 | 必填 |
| `--calibration` | 相机标定 YAML 路径 | 必填 |
| `--output` | 输出 JSONL 路径 | 必填 |
| `--confidence` | objectness 置信度阈值 | 0.5 |
| `--nms-iou` | NMS IoU 阈值 | 0.5 |
| `--max-frames` | 可选的调试帧数限制 | 不限制 |

## 验证结果

- `small.mp4`：360 帧
- `large.mp4`：360 帧
- JSON 格式、连续帧号和时间戳检查通过
- 二维与三维角点结构检查通过
- 输出中没有 NaN 或 Infinity
- 有效三维角点均位于相机前方

## 主要文件

- `detection/detection.py`：路线 A 主程序
- `detection/requirements.txt`：Python 依赖
- `detection/assets/`：教师提供的模型、录像和相机标定

本实现不根据视频文件名、帧范围或目标颜色硬编码检测结果，也不进行跨帧跟踪。
