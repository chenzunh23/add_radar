"""
SMPL格式joblib数据可视化工具
支持Z轴作为垂直方向的SMPL运动数据
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import os

def setup_chinese_font():
    """设置中文字体"""
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

# SMPL关节连接关系 (45关节扩展版本)
SMPL_JOINT_CONNECTIONS = [
    # 脊柱连接
    (0, 1),   # Pelvis -> L_Hip
    (0, 2),   # Pelvis -> R_Hip  
    (0, 3),   # Pelvis -> Spine1
    (3, 6),   # Spine1 -> Spine2
    (6, 9),   # Spine2 -> Spine3
    (9, 12),  # Spine3 -> Neck
    (12, 15), # Neck -> Head
    
    # 左腿
    (1, 4),   # L_Hip -> L_Knee
    (4, 7),   # L_Knee -> L_Ankle
    (7, 10),  # L_Ankle -> L_Foot
    
    # 右腿
    (2, 5),   # R_Hip -> R_Knee
    (5, 8),   # R_Knee -> R_Ankle
    (8, 11),  # R_Ankle -> R_Foot
    
    # 左肩膀和手臂
    (9, 13),  # Spine3 -> L_Collar
    (13, 16), # L_Collar -> L_Shoulder
    (16, 18), # L_Shoulder -> L_Elbow
    (18, 20), # L_Elbow -> L_Wrist
    (20, 22), # L_Wrist -> L_Hand
    
    # 右肩膀和手臂
    (9, 14),  # Spine3 -> R_Collar
    (14, 17), # R_Collar -> R_Shoulder
    (17, 19), # R_Shoulder -> R_Elbow
    (19, 21), # R_Elbow -> R_Wrist
    (21, 23), # R_Wrist -> R_Hand
]

# SMPL关节名称 (45关节)
SMPL_JOINT_NAMES = [
    'Pelvis',       # 0
    'L_Hip',        # 1
    'R_Hip',        # 2
    'Spine1',       # 3
    'L_Knee',       # 4
    'R_Knee',       # 5
    'Spine2',       # 6
    'L_Ankle',      # 7
    'R_Ankle',      # 8
    'Spine3',       # 9
    'L_Foot',       # 10
    'R_Foot',       # 11
    'Neck',         # 12
    'L_Collar',     # 13
    'R_Collar',     # 14
    'Head',         # 15
    'L_Shoulder',   # 16
    'R_Shoulder',   # 17
    'L_Elbow',      # 18
    'R_Elbow',      # 19
    'L_Wrist',      # 20
    'R_Wrist',      # 21
    'L_Hand',       # 22
    'R_Hand',       # 23
    # 手指关节 (24-44)
    'L_Thumb1', 'L_Thumb2', 'L_Thumb3',     # 24-26
    'L_Index1', 'L_Index2', 'L_Index3',     # 27-29
    'L_Middle1', 'L_Middle2', 'L_Middle3',  # 30-32
    'L_Ring1', 'L_Ring2', 'L_Ring3',        # 33-35
    'L_Pinky1', 'L_Pinky2', 'L_Pinky3',     # 36-38
    'R_Thumb1', 'R_Thumb2', 'R_Thumb3',     # 39-41
    'R_Index1', 'R_Index2', 'R_Index3',     # 42-44
]

def load_joblib_data(filename):
    """加载joblib格式数据"""
    try:
        data = joblib.load(filename)
        print(f"✓ 加载成功: {filename}")
        
        # 检查数据结构
        if isinstance(data, dict):
            # 查找包含实际数据的键
            for key in data.keys():
                if key.endswith('.pkl'):
                    actual_data = data[key]
                    if 'smpl_joints' in actual_data:
                        return actual_data
            print("未找到包含smpl_joints的数据")
            return None
        else:
            print(f"数据格式不正确: {type(data)}")
            return None
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def visualize_skeleton_3d(joints, title="SMPL骨架", frame_idx=0):
    """3D可视化SMPL骨架"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取当前帧的关节位置
    if joints.ndim == 3:  # (frames, joints, 3)
        frame_joints = joints[frame_idx]
    else:  # (joints, 3)
        frame_joints = joints
    
    # 绘制关节点
    ax.scatter(frame_joints[:, 0], frame_joints[:, 1], frame_joints[:, 2], 
               c='red', s=50, alpha=0.8, label='关节')
    
    # 绘制骨骼连接
    for connection in SMPL_JOINT_CONNECTIONS:
        if connection[0] < len(frame_joints) and connection[1] < len(frame_joints):
            start = frame_joints[connection[0]]
            end = frame_joints[connection[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   'b-', linewidth=2, alpha=0.7)
    
    # 标注重要关节
    important_joints = [0, 12, 15, 10, 11]  # Pelvis, Neck, Head, Feet
    for i in important_joints:
        if i < len(frame_joints):
            ax.text(frame_joints[i, 0], frame_joints[i, 1], frame_joints[i, 2], 
                   SMPL_JOINT_NAMES[i], fontsize=8)
    
    # 设置坐标轴（Z轴为垂直方向）
    ax.set_xlabel('X (左右)')
    ax.set_ylabel('Y (前后)')
    ax.set_zlabel('Z (上下)')
    
    # 统一坐标轴比例
    max_range = np.max([
        np.max(frame_joints[:, 0]) - np.min(frame_joints[:, 0]),
        np.max(frame_joints[:, 1]) - np.min(frame_joints[:, 1]),
        np.max(frame_joints[:, 2]) - np.min(frame_joints[:, 2])
    ])
    
    center = np.mean(frame_joints, axis=0)
    ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
    ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
    ax.set_zlim([center[2] - max_range/2, center[2] + max_range/2])
    
    ax.set_title(f'{title} - 帧{frame_idx}')
    ax.legend()
    
    return fig, ax

def analyze_motion_stats(data):
    """分析运动统计信息"""
    joints = data['smpl_joints']  # (frames, 45, 3)
    fps = data.get('fps', 30)
    
    print(f"=== 运动分析 ===")
    print(f"帧数: {joints.shape[0]}")
    print(f"关节数: {joints.shape[1]}")
    print(f"坐标维度: {joints.shape[2]}")
    print(f"FPS: {fps}")
    print(f"时长: {joints.shape[0]/fps:.2f}秒")
    
    # 坐标范围分析
    print(f"\n数据范围:")
    print(f"  X: [{np.min(joints[:, :, 0]):.3f}, {np.max(joints[:, :, 0]):.3f}]")
    print(f"  Y: [{np.min(joints[:, :, 1]):.3f}, {np.max(joints[:, :, 1]):.3f}]")
    print(f"  Z: [{np.min(joints[:, :, 2]):.3f}, {np.max(joints[:, :, 2]):.3f}]")
    
    # 双脚分析
    if joints.shape[1] >= 12:  # 确保有足够的关节
        left_foot = joints[:, 10, :]   # L_Foot
        right_foot = joints[:, 11, :]  # R_Foot
        foot_height_diff = np.abs(left_foot[:, 2] - right_foot[:, 2])
        
        print(f"\n双脚高度差异:")
        print(f"  最大差异: {np.max(foot_height_diff):.3f}m")
        print(f"  平均差异: {np.mean(foot_height_diff):.3f}m")
    
    # 根部运动分析
    pelvis = joints[:, 0, :]  # Pelvis
    pelvis_velocity = np.linalg.norm(np.diff(pelvis, axis=0), axis=1)
    
    print(f"\n骨盆移动:")
    print(f"  平均速度: {np.mean(pelvis_velocity):.4f}m/帧")
    print(f"  最大速度: {np.max(pelvis_velocity):.4f}m/帧")

def plot_motion_trajectory(data):
    """绘制运动轨迹"""
    joints = data['smpl_joints']  # (frames, 45, 3)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 骨盆轨迹
    pelvis = joints[:, 0, :]  # Pelvis
    axes[0, 0].plot(pelvis[:, 0], pelvis[:, 1])
    axes[0, 0].set_title('骨盆轨迹 (俯视图: X-Y)')
    axes[0, 0].set_xlabel('X (左右)')
    axes[0, 0].set_ylabel('Y (前后)')
    axes[0, 0].grid(True)
    
    # 高度变化
    axes[0, 1].plot(pelvis[:, 2], label='骨盆高度')
    if joints.shape[1] >= 12:
        axes[0, 1].plot(joints[:, 10, 2], label='左脚高度')
        axes[0, 1].plot(joints[:, 11, 2], label='右脚高度')
    axes[0, 1].set_title('高度变化 (Z轴)')
    axes[0, 1].set_xlabel('帧数')
    axes[0, 1].set_ylabel('高度 (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 速度分析
    pelvis_velocity = np.linalg.norm(np.diff(pelvis, axis=0), axis=1)
    axes[1, 0].plot(pelvis_velocity)
    axes[1, 0].set_title('骨盆运动速度')
    axes[1, 0].set_xlabel('帧数')
    axes[1, 0].set_ylabel('速度 (m/帧)')
    axes[1, 0].grid(True)
    
    # 双脚高度差异
    if joints.shape[1] >= 12:
        foot_height_diff = np.abs(joints[:, 10, 2] - joints[:, 11, 2])
        axes[1, 1].plot(foot_height_diff)
        axes[1, 1].set_title('双脚高度差异')
        axes[1, 1].set_xlabel('帧数')
        axes[1, 1].set_ylabel('高度差异 (m)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    setup_chinese_font()
    
    # 支持命令行参数指定文件
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "test.pkl"
    
    # 加载数据
    data = load_joblib_data(filename)
    if data is None:
        print("数据加载失败")
        return
    
    # 分析运动统计
    analyze_motion_stats(data)
    
    # 3D骨架可视化
    joints = data['smpl_joints']
    fig1, ax1 = visualize_skeleton_3d(joints, f"SMPL骨架 - {os.path.basename(filename)}", frame_idx=0)
    
    # 运动轨迹分析
    fig2 = plot_motion_trajectory(data)
    
    plt.show()

if __name__ == "__main__":
    main()
