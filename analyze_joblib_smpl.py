#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用joblib SMPL数据可视化工具
支持任何joblib格式的SMPL文件
"""
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_joblib_file(filename):
    """分析joblib文件结构"""
    print(f"=== 分析文件: {filename} ===")
    
    try:
        data = joblib.load(filename)
        print("✓ 加载成功")
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"顶层键: {list(data.keys())}")
            
            # 查找嵌套数据
            nested_data = None
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"\n{key} 内容:")
                    nested_data = value
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            print(f"  {subkey}: {subvalue.shape} {subvalue.dtype}")
                        else:
                            print(f"  {subkey}: {type(subvalue).__name__} = {subvalue}")
            
            return nested_data if nested_data else data
        
        return data
        
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None

def visualize_smpl_motion(data, target_frames=None):
    """可视化SMPL运动数据"""
    if 'smpl_joints' not in data:
        print("✗ 未找到smpl_joints数据")
        return
    
    smpl_joints = data['smpl_joints']
    num_frames, num_joints, coords = smpl_joints.shape
    
    print(f"\n=== SMPL运动分析 ===")
    print(f"帧数: {num_frames}")
    print(f"关节数: {num_joints}")
    print(f"坐标维度: {coords}")
    
    # 数据范围
    print(f"数据范围:")
    print(f"  X: [{smpl_joints[:, :, 0].min():.3f}, {smpl_joints[:, :, 0].max():.3f}]")
    print(f"  Y: [{smpl_joints[:, :, 1].min():.3f}, {smpl_joints[:, :, 1].max():.3f}]") 
    print(f"  Z: [{smpl_joints[:, :, 2].min():.3f}, {smpl_joints[:, :, 2].max():.3f}]")
    
    # FPS和时长
    if 'fps' in data:
        fps = data['fps']
        duration = num_frames / fps
        print(f"  FPS: {fps}")
        print(f"  时长: {duration:.2f}秒")
    
    # 关键关节分析
    if num_joints >= 24:  # 基本SMPL关节
        print(f"\n=== 关键关节分析 ===")
        
        # 假设关节顺序：0=Pelvis, 10=L_Foot, 11=R_Foot
        pelvis_idx = 0
        if num_joints > 10:
            left_foot_idx = 10
            right_foot_idx = 11
            
            # 足部高度分析
            left_foot_z = smpl_joints[:, left_foot_idx, 2]
            right_foot_z = smpl_joints[:, right_foot_idx, 2]
            
            foot_diff = np.abs(left_foot_z - right_foot_z)
            max_diff = np.max(foot_diff)
            mean_diff = np.mean(foot_diff)
            
            print(f"双脚高度差异:")
            print(f"  最大差异: {max_diff:.3f}m")
            print(f"  平均差异: {mean_diff:.3f}m")
            
            # 骨盆移动分析
            pelvis_pos = smpl_joints[:, pelvis_idx, :]
            if num_frames > 1:
                pelvis_velocity = np.sqrt(np.sum(np.diff(pelvis_pos, axis=0)**2, axis=1))
                avg_velocity = np.mean(pelvis_velocity)
                max_velocity = np.max(pelvis_velocity)
                
                print(f"骨盆移动:")
                print(f"  平均速度: {avg_velocity:.4f}m/帧")
                print(f"  最大速度: {max_velocity:.4f}m/帧")
    
    # 确定要可视化的帧
    if target_frames is not None:
        # 验证帧索引有效性
        valid_frames = []
        for frame_idx in target_frames:
            if 0 <= frame_idx < num_frames:
                valid_frames.append(frame_idx)
            else:
                print(f"警告: 帧索引 {frame_idx} 超出范围 [0, {num_frames-1}]，已跳过")
        
        if not valid_frames:
            print("错误: 没有有效的帧索引")
            return
            
        key_frames = valid_frames
        print(f"可视化指定帧: {key_frames}")
    else:
        # 默认选择关键帧
        key_frames = [0, num_frames//2, num_frames-1] if num_frames > 2 else [0]
        print(f"可视化关键帧: {key_frames}")
    
    # 简单的3D可视化
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 5))
        
        # 定义SMPL关节名称和分类（基于45关节SMPL格式 + 雷达）
        smpl_joint_names = [
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
            'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
            'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
            'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
        ] + [f'Joint_{i}' for i in range(24, 45)] + ['Radar']  # 剩余关节 + 雷达
        
        # 定义关节分类和颜色（与embodied_ai_processor.py相同）
        def classify_joint(idx, name):
            limb_end_names = ['L_Foot', 'R_Foot', 'L_Hand', 'R_Hand']
            head_names = ['Head', 'Neck']
            root_names = ['Pelvis']
            radar_names = ['Radar']
            
            if name in root_names:
                return 'root'
            elif name in limb_end_names:
                return 'limb_end'  
            elif name in head_names:
                return 'head'
            elif name in radar_names:
                return 'radar'
            else:
                return 'normal'
        
        # 颜色配置（与embodied_ai_processor.py相同）
        joint_colors = {
            'root': {'color': 'green', 'size': 60, 'marker': 's', 'alpha': 0.9},
            'limb_end': {'color': 'orange', 'size': 40, 'marker': 'd', 'alpha': 0.8},
            'head': {'color': 'purple', 'size': 50, 'marker': '*', 'alpha': 0.8},
            'radar': {'color': 'red', 'size': 80, 'marker': '^', 'alpha': 0.9},
            'normal': {'color': 'blue', 'size': 15, 'marker': 'o', 'alpha': 0.6}
        }
        
        for i, frame_idx in enumerate(key_frames):
            ax = fig.add_subplot(1, len(key_frames), i+1, projection='3d')
            
            frame_joints = smpl_joints[frame_idx]
            
            # 收集各类型关节用于绘制图例
            joint_types_found = set()
            type_positions = {'root': [], 'limb_end': [], 'head': [], 'radar': [], 'normal': []}
            
            # 按类型分组收集关节
            for joint_idx in range(len(frame_joints)):
                joint_name = smpl_joint_names[joint_idx] if joint_idx < len(smpl_joint_names) else f'Joint_{joint_idx}'
                joint_type = classify_joint(joint_idx, joint_name)
                joint_types_found.add(joint_type)
                
                pos = frame_joints[joint_idx]
                type_positions[joint_type].append(pos)
            
            # 按类型绘制关节点并添加图例标签
            type_labels = {
                'root': 'Root (Pelvis)',
                'limb_end': 'Limb Ends (Hands/Feet)',
                'head': 'Head (Head/Neck)',
                'radar': 'Radar Sensor',
                'normal': 'Normal Joints'
            }
            
            for joint_type in ['root', 'limb_end', 'head', 'radar', 'normal']:
                if joint_type in joint_types_found and type_positions[joint_type]:
                    config = joint_colors[joint_type]
                    positions = np.array(type_positions[joint_type])
                    
                    # 只在第一个子图显示图例标签
                    label = type_labels[joint_type] if i == 0 else None
                    
                    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                              c=config['color'], s=config['size'], marker=config['marker'], 
                              edgecolors='black', alpha=config['alpha'], 
                              label=label)
            
            # 绘制骨骼连接（主要连接）
            connections = [
                (0, 1), (0, 2), (0, 3),  # Pelvis connections
                (3, 6), (6, 9), (9, 12), (12, 15),  # Spine chain
                (1, 4), (4, 7), (7, 10),  # Left leg
                (2, 5), (5, 8), (8, 11),  # Right leg
                (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),  # Left arm
                (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),  # Right arm
            ]
            
            for start_idx, end_idx in connections:
                if start_idx < len(frame_joints) and end_idx < len(frame_joints):
                    start = frame_joints[start_idx]
                    end = frame_joints[end_idx]
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                           'gray', linewidth=1, alpha=0.6)
            
            # 设置坐标轴
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)') 
            ax.set_zlabel('Z (meters)')
            ax.set_title(f'Frame {frame_idx}')
            
            # 设置适当的显示范围（缩小范围以显示更大的人物）
            x_range = np.max(frame_joints[:, 0]) - np.min(frame_joints[:, 0])
            y_range = np.max(frame_joints[:, 1]) - np.min(frame_joints[:, 1])
            z_range = np.max(frame_joints[:, 2]) - np.min(frame_joints[:, 2])
            
            # 取最大范围的1.2倍作为显示范围，确保人物清晰可见
            max_range = max(x_range, y_range, z_range) * 1.2
            center = np.mean(frame_joints, axis=0)
            
            ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
            ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
            ax.set_zlim([center[2] - max_range/2, center[2] + max_range/2])
            
            # 添加完整图例（仅第一个子图）
            if i == 0:
                # 主图例
                legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)
                legend.set_title('Joint Types', prop={'weight': 'bold', 'size': 9})
                
            
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"可视化错误: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='通用joblib SMPL数据可视化工具')
    parser.add_argument('filename', nargs='?', default="lift_up_joblib.pkl", 
                       help='要分析的PKL文件路径')
    parser.add_argument('--frame', '-f', type=int, default=None,
                       help='指定要可视化的帧索引（默认显示关键帧）')
    parser.add_argument('--frames', '-fs', nargs='+', type=int, default=None,
                       help='指定多个要可视化的帧索引')
    
    args = parser.parse_args()
    
    print(f"分析文件: {args.filename}")
    
    # 分析文件结构
    data = analyze_joblib_file(args.filename)
    
    if data:
        # 可视化运动数据
        if args.frame is not None:
            # 单帧模式
            visualize_smpl_motion(data, target_frames=[args.frame])
        elif args.frames is not None:
            # 多帧模式
            visualize_smpl_motion(data, target_frames=args.frames)
        else:
            # 默认关键帧模式
            visualize_smpl_motion(data)
    else:
        print("无法分析文件")

if __name__ == "__main__":
    main()
