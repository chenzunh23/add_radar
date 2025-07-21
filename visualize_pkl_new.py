# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib的中文字体"""
    try:
        import matplotlib.font_manager as fm
        
        # 常见的中文字体名称
        font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 
                     'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK']
        
        for font_name in font_names:
            try:
                font_prop = fm.FontProperties(family=font_name)
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用中文字体: {font_name}")
                return font_name
            except:
                continue
        
        # 如果没有找到中文字体，使用默认设置
        print("警告: 未找到中文字体，使用默认字体")
        plt.rcParams['axes.unicode_minus'] = False
        return None
    except Exception as e:
        print(f"字体设置错误: {e}")
        return None

def classify_joints(joint_names):
    """将关节分类为不同类型"""
    # 四肢末端关节
    limb_end_names = [
        'LeftHand', 'RightHand', 'LeftToeBase', 'RightToeBase', 
        'LeftFoot', 'RightFoot', 'LeftToe_End', 'RightToe_End'
    ]
    
    # 头部关节
    head_names = ['Head', 'Neck', 'Neck1']
    
    # 根节点
    root_names = ['Hips', 'Hip']
    
    # 雷达关节
    radar_names = ['Mid360Radar', 'radar', 'Radar']
    
    classification = {}
    for i, name in enumerate(joint_names):
        if name in root_names:
            classification[i] = 'root'
        elif name in radar_names:
            classification[i] = 'radar'
        elif name in limb_end_names:
            classification[i] = 'limb_end'
        elif name in head_names:
            classification[i] = 'head'
        else:
            classification[i] = 'normal'
    
    return classification

def detect_outliers(positions, threshold_factor=3):
    """检测异常关节位置"""
    # 计算所有关节到中心的距离
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    
    # 使用统计阈值检测异常值
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    threshold = median_dist + threshold_factor * std_dist
    
    outliers = distances > threshold
    return outliers, distances, center

def visualize_frame(data, frame_idx=0):
    """可视化单帧数据"""
    setup_chinese_font()
    
    joint_names = data['joint_names']
    positions = data['joint_positions'][frame_idx]
    joint_parents = data['joint_parents']
    
    # 关节分类
    classification = classify_joints(joint_names)
    
    # 异常值检测
    outliers, distances, center = detect_outliers(positions)
    
    # 创建4个子图
    fig = plt.figure(figsize=(20, 15))
    
    # 3D视图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # 绘制关节连接线
    for i, joint_name in enumerate(joint_names):
        parent_name = joint_parents.get(joint_name)
        if parent_name and parent_name in joint_names:
            parent_idx = joint_names.index(parent_name)
            joint_pos = positions[i]
            parent_pos = positions[parent_idx]
            
            ax1.plot3D([parent_pos[0], joint_pos[0]], 
                      [parent_pos[1], joint_pos[1]], 
                      [parent_pos[2], joint_pos[2]], 'gray', alpha=0.5)
    
    # 按分类绘制关节
    colors = {'root': 'green', 'radar': 'red', 'limb_end': 'orange', 'head': 'purple', 'normal': 'blue'}
    markers = {'root': 's', 'radar': '^', 'limb_end': 'D', 'head': '*', 'normal': 'o'}
    sizes = {'root': 100, 'radar': 150, 'limb_end': 80, 'head': 100, 'normal': 30}
    
    for joint_type in colors.keys():
        indices = [i for i, t in classification.items() if t == joint_type]
        if indices:
            type_positions = positions[indices]
            ax1.scatter(type_positions[:, 0], type_positions[:, 1], type_positions[:, 2],
                       c=colors[joint_type], s=sizes[joint_type], marker=markers[joint_type],
                       label=f'{joint_type} ({len(indices)})')
    
    # 标记异常关节
    outlier_indices = np.where(outliers)[0]
    if len(outlier_indices) > 0:
        outlier_positions = positions[outlier_indices]
        ax1.scatter(outlier_positions[:, 0], outlier_positions[:, 1], outlier_positions[:, 2],
                   c='black', s=200, marker='x', linewidth=3, label=f'异常点 ({len(outlier_indices)})')
    
    ax1.set_xlabel('X坐标 (cm)')
    ax1.set_ylabel('Y坐标 (cm)')
    ax1.set_zlabel('Z坐标 (cm)')
    ax1.legend()
    ax1.set_title(f'3D骨架可视化 - 第{frame_idx}帧')
    
    # XY平面视图
    ax2 = fig.add_subplot(2, 2, 2)
    for joint_type in colors.keys():
        indices = [i for i, t in classification.items() if t == joint_type]
        if indices:
            type_positions = positions[indices]
            ax2.scatter(type_positions[:, 0], type_positions[:, 1],
                       c=colors[joint_type], s=sizes[joint_type], marker=markers[joint_type],
                       alpha=0.7, label=f'{joint_type}')
    
    if len(outlier_indices) > 0:
        outlier_positions = positions[outlier_indices]
        ax2.scatter(outlier_positions[:, 0], outlier_positions[:, 1],
                   c='black', s=200, marker='x', linewidth=3, label='异常点')
    
    ax2.set_xlabel('X坐标 (cm)')
    ax2.set_ylabel('Y坐标 (cm)')
    ax2.set_title('XY平面投影')
    ax2.grid(True, alpha=0.3)
    
    # XZ平面视图
    ax3 = fig.add_subplot(2, 2, 3)
    for joint_type in colors.keys():
        indices = [i for i, t in classification.items() if t == joint_type]
        if indices:
            type_positions = positions[indices]
            ax3.scatter(type_positions[:, 0], type_positions[:, 2],
                       c=colors[joint_type], s=sizes[joint_type], marker=markers[joint_type],
                       alpha=0.7, label=f'{joint_type}')
    
    if len(outlier_indices) > 0:
        outlier_positions = positions[outlier_indices]
        ax3.scatter(outlier_positions[:, 0], outlier_positions[:, 2],
                   c='black', s=200, marker='x', linewidth=3, label='异常点')
    
    ax3.set_xlabel('X坐标 (cm)')
    ax3.set_ylabel('Z坐标 (cm)')
    ax3.set_title('XZ平面投影')
    ax3.grid(True, alpha=0.3)
    
    # YZ平面视图
    ax4 = fig.add_subplot(2, 2, 4)
    for joint_type in colors.keys():
        indices = [i for i, t in classification.items() if t == joint_type]
        if indices:
            type_positions = positions[indices]
            ax4.scatter(type_positions[:, 1], type_positions[:, 2],
                       c=colors[joint_type], s=sizes[joint_type], marker=markers[joint_type],
                       alpha=0.7, label=f'{joint_type}')
    
    if len(outlier_indices) > 0:
        outlier_positions = positions[outlier_indices]
        ax4.scatter(outlier_positions[:, 1], outlier_positions[:, 2],
                   c='black', s=200, marker='x', linewidth=3, label='异常点')
    
    ax4.set_xlabel('Y坐标 (cm)')
    ax4.set_ylabel('Z坐标 (cm)')
    ax4.set_title('YZ平面投影')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印分析报告
    print("\n=== 关节分析报告 ===")
    for joint_type, color in colors.items():
        indices = [i for i, t in classification.items() if t == joint_type]
        if indices:
            print(f"{joint_type}: {len(indices)}个关节")
            for idx in indices:
                joint_name = joint_names[idx]
                pos = positions[idx]
                dist = distances[idx]
                outlier_mark = " [异常]" if outliers[idx] else ""
                print(f"  {joint_name}: 位置=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) cm, 距中心={dist:.1f} cm{outlier_mark}")
    
    if len(outlier_indices) > 0:
        print(f"\n检测到 {len(outlier_indices)} 个异常关节:")
        for idx in outlier_indices:
            joint_name = joint_names[idx]
            pos = positions[idx]
            dist = distances[idx]
            print(f"  {joint_name}: 距中心 {dist:.1f} cm")

def analyze_radar_movement(data):
    """分析雷达运动轨迹"""
    if 'mid360_radar_index' not in data:
        print("未找到雷达关节")
        return
    
    radar_idx = data['mid360_radar_index']
    radar_positions = data['joint_positions'][:, radar_idx, :]
    
    print("\n雷达关节分析:")
    print(f"  位置范围 X: [{np.min(radar_positions[:, 0]):.2f}, {np.max(radar_positions[:, 0]):.2f}] cm")
    print(f"  位置范围 Y: [{np.min(radar_positions[:, 1]):.2f}, {np.max(radar_positions[:, 1]):.2f}] cm")
    print(f"  位置范围 Z: [{np.min(radar_positions[:, 2]):.2f}, {np.max(radar_positions[:, 2]):.2f}] cm")
    
    # 计算运动速度
    if len(radar_positions) > 1:
        diff = np.diff(radar_positions, axis=0)
        speed = np.linalg.norm(diff, axis=1) / data['frame_time']
        print(f"  平均速度: {np.mean(speed):.2f} cm/s")
        print(f"  最大速度: {np.max(speed):.2f} cm/s")
    
    # 相对于Hips的偏移
    joint_names = data['joint_names']
    if 'Hips' in joint_names:
        hips_idx = joint_names.index('Hips')
        hips_positions = data['joint_positions'][:, hips_idx, :]
        relative_pos = radar_positions - hips_positions
        print(f"  相对Hips的平均偏移: [{np.mean(relative_pos[:, 0]):.2f}, {np.mean(relative_pos[:, 1]):.2f}, {np.mean(relative_pos[:, 2]):.2f}] cm")

def main():
    import sys
    if len(sys.argv) != 2:
        print("用法: python visualize_pkl_new.py <pkl文件路径>")
        return
    
    pkl_file = sys.argv[1]
    
    print(f"加载PKL文件: {pkl_file}")
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        print("PKL文件内容:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)} (长度: {len(value) if hasattr(value, '__len__') else 'N/A'})")
        
        print(f"\n可视化第 0 帧 (共 {data['num_frames']} 帧)")
        visualize_frame(data, 0)
        analyze_radar_movement(data)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
