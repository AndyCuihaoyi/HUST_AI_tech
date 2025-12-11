# 轻量化YOLOv8视频检测（仅视频、纯CPU）
from ultralytics import YOLO
import cv2
import time
import os
from pathlib import Path
def yolo_video_detect(video_path, output_path=""):
    if not output_path:  # 若未手动指定输出路径，则自动生成
        # 解析原视频路径的文件名和上级目录
        video_path_obj = Path(video_path)
        # 提取原视频文件名（如test1.mp4）
        video_filename = video_path_obj.name
        # 拼接新路径：result/原文件名（自动创建result目录）
        output_dir = "result"
        os.makedirs(output_dir, exist_ok=True)  # 确保result目录存在
        output_path = os.path.join(output_dir, video_filename)
    # 1. 加载YOLOv8n轻量化模型（强制CPU运行）
    model = YOLO('yolov8n.pt')
    model.to('cpu')  # 显式指定CPU设备

    # 2. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")

    # 3. 获取视频基础参数（用于保存结果）
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出视频编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4. 初始化性能统计参数
    frame_count = 0
    total_infer_time = 0.0
    print(f"开始检测视频：{video_path} | 分辨率：{width}×{height} | 帧率：{fps}")

    # 5. 逐帧检测
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 计时：推理耗时
        start_time = time.time()
        # 核心检测：轻量化配置（低分辨率、高置信度过滤）
        results = model(
            frame,
            imgsz=320,        # 轻量化分辨率（越小越快）
            conf=0.25,        # 置信度阈值（过滤低精度结果）
            iou=0.45,         # 重叠框过滤阈值
            verbose=False     # 关闭冗余日志
        )
        infer_time = time.time() - start_time
        total_infer_time += infer_time
        frame_count += 1

        # 6. 标注检测结果到帧画面
        annotated_frame = results[0].plot()  # 自动绘制检测框、类别、置信度

        # 7. 写入结果视频 + 实时显示
        out.write(annotated_frame)
        cv2.imshow('YOLOv8 Video Detection (CPU)', annotated_frame)

        # 按q键提前退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 9. 输出性能统计
    avg_infer_time = total_infer_time / frame_count if frame_count > 0 else 0
    print("\n===== 检测完成 =====")
    print(f"总检测帧数：{frame_count}")
    print(f"单帧平均推理耗时：{avg_infer_time:.4f} 秒（CPU）")
    print(f"平均检测帧率：{1/avg_infer_time:.2f} FPS（CPU）")
    print(f"标注结果保存至：{output_path}")

if __name__ == '__main__':
    # 替换为你的视频路径（支持mp4/avi/mov等常见格式）
    VIDEO_PATH = "test_vedio/hust_yolo_test3.mp4"  # 本地视频文件路径
    yolo_video_detect(VIDEO_PATH)