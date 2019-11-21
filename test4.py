import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Configure depth and color streams
pipeline = rs.pipeline()
# 創建配置
config = rs.config()
# (攝影機，大小，串流模式，FPS)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
# (攝影機，大小，串流模式(YUYV、BGR8、RGBA8、BGRA8、Y16、RGB8)，FPS)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# 開始串流
pipeline.start(config)

try:
    # 串流迴圈

    while True:

        # Wait for a coherent pair of frames: depth and color
        # 等待一對連貫的幀：深度和顏色
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 轉換圖片成 numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # 在深度圖像上應用色彩圖（圖像必須先轉換為每像素8位）
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        # 水平堆疊兩個圖像
        images = np.hstack((color_image, depth_colormap))
        # images1 = np.hstack((color_image,depth_image))

        # 顯示圖片
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        cv2.imshow('RealSense', images)
        images1 = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        # plt.imshow(images1)
        # plt.axis("off")
        # plt.colorbar()

        cv2.imshow('RealSense_depth', depth_image)
        # #設置關閉按鍵
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            plt.close('all')
            cv2.destroyAllWindows()
            break

finally:
    # 串流關閉
    pipeline.stop()
