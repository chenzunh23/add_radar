"""
具身智能BVH到PKL转换工具
支持添加Mid360雷达关节，并提供完整的可视化功能

使用方法:
python embodied_ai_processor.py --input walk.bvh --output walk_with_radar.pkl --radar-offset 0.003 46.018 -0.368
"""

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import re

class EmbodiedAIProcessor:
    def __init__(self, radar_offset=None):
        """
        初始化处理器
        
        A        print(f"数据范围: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], Z[{z_min:.1f}, {z_max:.1f}]")
        print(f"统一范围: {max_range:.1f} mm, 中心: ({x_center:.1f}, {y_center:.1f}, {z_center:.1f})")
        
        # 过滤异常关节点（距离中心超过3倍标准差的点）
        distances_from_center = np.sqrt((positions[:, 0] - x_center)**2 + 
                                       (positions[:, 1] - y_center)**2 + 
                                       (positions[:, 2] - z_center)**2)
        median_distance = np.median(distances_from_center)
        std_distance = np.std(distances_from_center)
        threshold = median_distance + 3 * std_distance
        
        # 标记异常点
        outlier_mask = distances_from_center > threshold
        normal_mask = ~outlier_mask
        
        if np.any(outlier_mask):
            outlier_indices = np.where(outlier_mask)[0]
            print(f"检测到 {len(outlier_indices)} 个异常关节点:")
            for idx in outlier_indices:
                joint_name = joint_names[idx] if idx < len(joint_names) else f"关节{idx}"
                pos = positions[idx]
                print(f"  {joint_name}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        
        # 重新计算正常关节的范围用于可视化
        if np.any(normal_mask):
            normal_positions = positions[normal_mask]
            x_min_n, x_max_n = np.min(normal_positions[:, 0]), np.max(normal_positions[:, 0])
            y_min_n, y_max_n = np.min(normal_positions[:, 1]), np.max(normal_positions[:, 1])
            z_min_n, z_max_n = np.min(normal_positions[:, 2]), np.max(normal_positions[:, 2])
            
            # 计算正常关节的范围
            x_range_n = x_max_n - x_min_n
            y_range_n = y_max_n - y_min_n
            z_range_n = z_max_n - z_min_n
            max_range_n = max(x_range_n, y_range_n, z_range_n)
            
            # 使用正常关节的中心点
            x_center_n = (x_min_n + x_max_n) / 2
            y_center_n = (y_min_n + y_max_n) / 2
            z_center_n = (z_min_n + z_max_n) / 2
            
            # 如果正常范围更合理，使用正常范围
            if max_range_n < max_range * 0.8:  # 正常范围明显小于全部范围
                print(f"使用正常关节范围: {max_range_n:.1f} mm")
                x_center, y_center, z_center = x_center_n, y_center_n, z_center_n
                max_range = max_range_ns:
            radar_offset: 雷达相对于Hips的偏移 [x, y, z] (cm)
        """
        self.joint_names = []
        self.joint_offsets = {}
        self.joint_parents = {}
        self.joint_channels = {}
        self.motion_data = []
        self.frame_time = 0.0
        self.num_frames = 0
        
        # 默认雷达偏移位置 (相对于机器人Hips)
        if radar_offset is None:
            self.radar_offset = np.array([-0.368, 0.003, 46.018])  # X, Y, Z偏移(cm)
        else:
            self.radar_offset = np.array(radar_offset)
        
        print(f"雷达偏移设置: {self.radar_offset}")
        
    def parse_bvh(self, bvh_file_path):
        """解析BVH文件，提取骨架结构和运动数据"""
        print(f"正在解析BVH文件: {bvh_file_path}")
        
        if not os.path.exists(bvh_file_path):
            raise FileNotFoundError(f"BVH文件不存在: {bvh_file_path}")
        
        with open(bvh_file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        current_joint = None
        joint_stack = []
        
        # 解析骨架结构
        in_end_site = False
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('ROOT') or line.startswith('JOINT'):
                joint_name = line.split()[1]
                self.joint_names.append(joint_name)
                
                # 设置父关节
                if joint_stack:
                    self.joint_parents[joint_name] = joint_stack[-1]
                else:
                    self.joint_parents[joint_name] = None
                
                joint_stack.append(joint_name)
                current_joint = joint_name
                in_end_site = False
                
            elif line.startswith('OFFSET'):
                if current_joint and not in_end_site:
                    offset = [float(x) for x in line.split()[1:4]]
                    self.joint_offsets[current_joint] = np.array(offset)
                    
            elif line.startswith('CHANNELS'):
                if current_joint and not in_end_site:
                    parts = line.split()
                    num_channels = int(parts[1])
                    channels = parts[2:]
                    self.joint_channels[current_joint] = channels[:num_channels]
                    
            elif line.startswith('End Site'):
                in_end_site = True
                    
            elif line == '}':
                if in_end_site:
                    # End Site 结束
                    in_end_site = False
                else:
                    # 关节结束，弹出栈
                    if joint_stack:
                        joint_stack.pop()
                    current_joint = joint_stack[-1] if joint_stack else None
                    
            elif line.startswith('MOTION'):
                i += 1
                break
                
            i += 1
        
        # 解析运动数据
        if i < len(lines):
            frames_line = lines[i].strip()
            if frames_line.startswith('Frames:'):
                self.num_frames = int(frames_line.split()[1])
                i += 1
                
            frame_time_line = lines[i].strip()
            if frame_time_line.startswith('Frame Time:'):
                self.frame_time = float(frame_time_line.split()[2])
                i += 1
            
            # 读取运动数据
            for frame_idx in range(self.num_frames):
                if i + frame_idx < len(lines):
                    frame_data = [float(x) for x in lines[i + frame_idx].split()]
                    self.motion_data.append(frame_data)
        
        self.motion_data = np.array(self.motion_data)
        print(f"解析完成: {len(self.joint_names)} 个关节, {self.num_frames} 帧, 帧时间: {self.frame_time}s")
        # print(f"主要关节: {self.joint_names[:10]}...")  # 显示前10个关节
        
        # 调试：验证父子关节关系
        # print(f"\n父子关节关系验证:")
        # for i, joint_name in enumerate(self.joint_names[:15]):  # 显示前15个关节的关系
        #     parent = self.joint_parents.get(joint_name, 'None')
        #     offset = self.joint_offsets.get(joint_name, np.zeros(3))
        #     channels = self.joint_channels.get(joint_name, [])
        #     print(f"  {joint_name}: 父={parent}, 偏移={offset}, 通道={len(channels)}个")
        
    def add_mid360_radar_joint(self, parent_joint):
        """添加Mid360雷达关节到骨架结构中"""
        radar_joint_name = "Mid360Radar"
        
        # 检查是否已经存在雷达关节
        if radar_joint_name in self.joint_names:
            print(f"雷达关节 {radar_joint_name} 已存在，跳过添加")
            return
            
        # 添加雷达关节到关节列表
        self.joint_names.append(radar_joint_name)
        
        # # 设置雷达关节的父关节为Hips(pelvis)
        # parent_joint = None
        # for candidate in ["Hips", "Pelvis", "pelvis", "hips"]:
        #     if candidate in self.joint_names:
        #         parent_joint = candidate
        #         break
        
        if parent_joint is None:
            # 如果没有找到Hips，使用第一个关节作为父关节
            parent_joint = self.joint_names[0]
            print(f"警告: 未找到Hips关节，使用 {parent_joint} 作为雷达关节的父关节")
        
        self.joint_parents[radar_joint_name] = parent_joint
        
        # 设置雷达关节的偏移
        # 计算父关节相对根关节的偏移
        if parent_joint not in self.joint_offsets:
            print(f"警告: 父关节 {parent_joint} 没有偏移数据，使用默认偏移")

        parent_offset = np.zeros(3)
        parent = parent_joint
        while parent != "Hips" and parent is not None:
            parent_offset += self.joint_offsets.get(parent, np.zeros(3))
            parent = self.joint_parents.get(parent, None)
            # print(f" 关节{parent_joint} 相对于关节{parent} 的偏移: {parent_offset}")


        radar_offset = self.radar_offset - parent_offset
        self.joint_offsets[radar_joint_name] = radar_offset
        
        # 雷达关节通常只有位置，没有旋转通道
        self.joint_channels[radar_joint_name] = []
        
        print(f"已添加雷达关节: {radar_joint_name}")
        print(f"  父关节: {parent_joint}")
        print(f"  偏移: {radar_offset}")

    def compute_joint_positions(self):
        """使用正向运动学计算所有关节在世界坐标系中的3D位置"""
        num_frames = len(self.motion_data)
        num_joints = len(self.joint_names)
        
        print(f"计算 {num_frames} 帧 x {num_joints} 关节的位置 (使用正向运动学)...")
        
        # 初始化关节位置数组 [frames, joints, 3]
        joint_positions = np.zeros((num_frames, num_joints, 3))
        
        for frame_idx in range(num_frames):
            frame_data = self.motion_data[frame_idx]
            
            # 解析每个关节的变换数据
            data_idx = 0
            joint_transforms = {}
            
            # 首先解析所有关节的局部变换
            for joint_name in self.joint_names:
                # if joint_name == "Mid360Radar":
                #     continue  # 雷达关节单独处理
                    
                channels = self.joint_channels.get(joint_name, [])
                
                # 初始化局部变换
                local_translation = np.zeros(3)
                local_rotation = np.zeros(3)
                
                # 解析通道数据
                for channel in channels:
                    if data_idx < len(frame_data):
                        value = frame_data[data_idx]
                        data_idx += 1
                        
                        if channel == 'Xposition':
                            local_translation[0] = value
                        elif channel == 'Yposition':
                            local_translation[1] = value
                        elif channel == 'Zposition':
                            local_translation[2] = value
                        elif channel == 'Xrotation':
                            local_rotation[0] = np.radians(value)
                        elif channel == 'Yrotation':
                            local_rotation[1] = np.radians(value)
                        elif channel == 'Zrotation':
                            local_rotation[2] = np.radians(value)
                
                joint_transforms[joint_name] = {
                    'local_translation': local_translation,
                    'local_rotation': local_rotation,
                    'offset': self.joint_offsets.get(joint_name, np.zeros(3))
                }
            
            # 使用正向运动学计算世界坐标位置
            joint_world_transforms = {}
            
            # 按照层次结构递归计算世界变换
            def compute_world_transform(joint_name):
                if joint_name in joint_world_transforms:
                    return joint_world_transforms[joint_name]
                
                # if joint_name == "Mid360Radar":
                #     # 雷达关节特殊处理 - 跟随父关节的旋转
                #     parent_name = self.joint_parents[joint_name]
                #     print(f"  关节{joint_name} 的父关节: {parent_name}")
                #     if parent_name and parent_name in self.joint_names:
                #         parent_transform = compute_world_transform(parent_name)
                #         # 雷达位置 = 父关节世界位置 + 经过父关节旋转后的雷达偏移
                #         radar_offset_rotated = self._rotate_vector(self.joint_offsets[joint_name], parent_transform['rotation'])
                #         world_position = parent_transform['position'] + radar_offset_rotated
                #         joint_world_transforms[joint_name] = {
                #             'position': world_position,
                #             'rotation': parent_transform['rotation']  # 雷达继承父关节旋转
                #         }
                #     else:
                #         # 如果没有父关节，使用原始偏移
                #         joint_world_transforms[joint_name] = {
                #             'position': self.radar_offset.copy(),
                #             'rotation': np.zeros(3)
                #         }
                #     return joint_world_transforms[joint_name]
                
                # 普通关节处理
                parent_name = self.joint_parents[joint_name]
                transform_data = joint_transforms.get(joint_name, {
                    'local_translation': np.zeros(3),
                    'local_rotation': np.zeros(3),
                    'offset': np.zeros(3)
                })
                
                if parent_name is None:
                    # 根关节
                    world_position = transform_data['local_translation'] + transform_data['offset']
                    world_rotation = transform_data['local_rotation']
                    
                    # 调试根节点位置计算
                    if joint_name == "Hips" and frame_idx == 50:
                        print(f"\n根节点调试 (第{frame_idx}帧):")
                        print(f"  原始运动数据前6项: {frame_data[:6]}")
                        print(f"  {joint_name} local_translation: {transform_data['local_translation']}")
                        print(f"  {joint_name} offset: {transform_data['offset']}")
                        print(f"  {joint_name} world_position: {world_position}")
                else:
                    # 子关节：继承父关节的变换
                    parent_transform = compute_world_transform(parent_name)
                    
                    # 计算世界旋转（父旋转 + 局部旋转）
                    world_rotation = parent_transform['rotation'] + transform_data['local_rotation']
                    
                    # 计算世界位置：父位置 + 旋转后的(偏移 + 局部位移)
                    local_offset = transform_data['offset'] + transform_data['local_translation']
                    rotated_offset = self._rotate_vector(local_offset, parent_transform['rotation'])
                    world_position = parent_transform['position'] + rotated_offset
                
                joint_world_transforms[joint_name] = {
                    'position': world_position,
                    'rotation': world_rotation
                }
                return joint_world_transforms[joint_name]
            
            # 计算所有关节的世界位置
            for joint_idx, joint_name in enumerate(self.joint_names):
                world_transform = compute_world_transform(joint_name)
                joint_positions[frame_idx, joint_idx] = world_transform['position']
        
        print("正向运动学计算完成")
        return joint_positions
    
    def _rotate_vector(self, vector, rotation):
        """使用欧拉角旋转向量 (按照BVH标准的ZXY顺序)"""
        if np.allclose(rotation, 0):
            return vector.copy()
        
        # BVH通常使用ZXY欧拉角顺序
        rx, ry, rz = rotation
        
        # 创建旋转矩阵
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        
        # Z旋转矩阵
        Rz = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        
        # X旋转矩阵
        Rx = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        
        # Y旋转矩阵
        Ry = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        # 组合旋转矩阵 (ZXY顺序)
        R = Ry @ Rx @ Rz
        
        return R @ vector
    
    def create_pkl_data(self, joint_positions):
        """创建PKL数据结构，包含所有必要信息"""
        
        # 查找雷达关节索引
        radar_idx = -1
        if "Mid360Radar" in self.joint_names:
            radar_idx = self.joint_names.index("Mid360Radar")
        
        data = {
            # 基本结构信息
            'joint_names': self.joint_names,
            'joint_positions': joint_positions,  # [frames, joints, 3]
            'joint_offsets': self.joint_offsets,
            'joint_parents': self.joint_parents,
            'joint_channels': self.joint_channels,
            
            # 时间信息
            'frame_time': self.frame_time,
            'num_frames': self.num_frames,
            
            # 雷达特定信息
            'mid360_radar_index': radar_idx,
            'mid360_radar_offset': self.radar_offset,
            'mid360_radar_parent': self.joint_parents.get("Mid360Radar", None),
            
            # 元数据
            'creation_timestamp': np.datetime64('now').astype(str),
            'total_joints': len(self.joint_names),
            'original_motion_data_shape': self.motion_data.shape,
        }
        
        print(f"\nPKL数据结构创建完成:")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, (list, dict)):
                print(f"  {key}: {type(value).__name__} (长度: {len(value)})")
            else:
                print(f"  {key}: {value}")
        
        return data
    
    def save_pkl(self, data, output_path):
        """保存PKL文件"""
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\nPKL文件已成功保存到: {output_path}")
            
            # 验证文件
            file_size = os.path.getsize(output_path)
            print(f"文件大小: {file_size:,} 字节")
            
            # 快速验证文件可读性
            with open(output_path, 'rb') as f:
                test_data = pickle.load(f)
            print("文件验证通过 ✓")
            
        except Exception as e:
            print(f"保存PKL文件时出错: {e}")
            raise
    
    def visualize_skeleton(self, data, frame_indices=None, save_plot=False):
        """可视化骨架结构（统一比例尺），支持指定多个帧"""
        joint_positions = data['joint_positions']
        joint_names = data['joint_names']
        joint_parents = data['joint_parents']
        
        # 确定要可视化的帧
        if frame_indices is not None:
            # 验证帧索引有效性
            valid_frames = []
            for frame_idx in frame_indices:
                if 0 <= frame_idx < len(joint_positions):
                    valid_frames.append(frame_idx)
                else:
                    print(f"警告: 帧索引 {frame_idx} 超出范围 [0, {len(joint_positions)-1}]，已跳过")
            
            if not valid_frames:
                print("错误: 没有有效的帧索引")
                return
                
            target_frames = valid_frames
            print(f"可视化指定帧: {target_frames}")
        else:
            # 默认只显示第一帧
            target_frames = [0]
            print(f"可视化默认帧: {target_frames}")
        
        # 设置图形大小，根据帧数调整
        num_frames_to_show = len(target_frames)
        if num_frames_to_show == 1:
            fig_size = (12, 10)
            subplot_layout = (2, 2)
        elif num_frames_to_show <= 3:
            fig_size = (16, 12)
            subplot_layout = (2, num_frames_to_show + 1)  # +1 for 3D view
        else:
            fig_size = (20, 15)
            subplot_layout = (3, min(4, num_frames_to_show))
        
        # 设置中文字体（解决中文显示问题）
        try:
            # 尝试设置中文字体
            import matplotlib.font_manager as fm
            
            # 查找系统中可用的中文字体
            font_list = fm.findSystemFonts()
            chinese_fonts = []
            
            # 常见的中文字体名称
            font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 
                         'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK']
            
            for font_name in font_names:
                try:
                    # 尝试创建字体属性
                    font_prop = fm.FontProperties(family=font_name)
                    chinese_fonts.append(font_name)
                    break
                except:
                    continue
            
            if chinese_fonts:
                plt.rcParams['font.sans-serif'] = chinese_fonts
                print(f"使用中文字体: {chinese_fonts[0]}")
            else:
                # 如果没有找到中文字体，使用默认字体并显示警告
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                print("警告: 未找到中文字体，中文可能显示为方块")
                
        except ImportError:
            # 如果无法导入font_manager，使用简单设置
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=fig_size)
        
        # 对每个指定的帧进行可视化
        for plot_idx, frame_idx in enumerate(target_frames):
            positions = joint_positions[frame_idx]
        
        # 计算数据范围，确保统一比例尺
        x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
        y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
        z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])
        
        # 计算最大范围，用于统一比例尺
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)
        
        # 计算中心点
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        # 添加一些边距
        margin = max_range * 0.1
        half_range = max_range / 2 + margin
        
        print(f"数据范围: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], Z[{z_min:.1f}, {z_max:.1f}]")
        print(f"统一范围: {max_range:.1f} cm, 中心: ({x_center:.1f}, {y_center:.1f}, {z_center:.1f})")
        
        # 过滤异常关节点（距离中心超过3倍标准差的点）
        distances_from_center = np.sqrt((positions[:, 0] - x_center)**2 + 
                                       (positions[:, 1] - y_center)**2 + 
                                       (positions[:, 2] - z_center)**2)
        median_distance = np.median(distances_from_center)
        std_distance = np.std(distances_from_center)
        threshold = median_distance + 3 * std_distance
        
        # 标记异常点
        outlier_mask = distances_from_center > threshold
        normal_mask = ~outlier_mask
        
        # if np.any(outlier_mask):
        #     outlier_indices = np.where(outlier_mask)[0]
        #     print(f"检测到 {len(outlier_indices)} 个异常关节点:")
        #     for idx in outlier_indices:
        #         joint_name = joint_names[idx] if idx < len(joint_names) else f"关节{idx}"
        #         pos = positions[idx]
        #         print(f"  {joint_name}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        
        # 重新计算正常关节的范围用于可视化
        if np.any(normal_mask):
            normal_positions = positions[normal_mask]
            x_min_n, x_max_n = np.min(normal_positions[:, 0]), np.max(normal_positions[:, 0])
            y_min_n, y_max_n = np.min(normal_positions[:, 1]), np.max(normal_positions[:, 1])
            z_min_n, z_max_n = np.min(normal_positions[:, 2]), np.max(normal_positions[:, 2])
            
            # 计算正常关节的范围
            x_range_n = x_max_n - x_min_n
            y_range_n = y_max_n - y_min_n
            z_range_n = z_max_n - z_min_n
            max_range_n = max(x_range_n, y_range_n, z_range_n)
            
            # 使用正常关节的中心点
            x_center_n = (x_min_n + x_max_n) / 2
            y_center_n = (y_min_n + y_max_n) / 2
            z_center_n = (z_min_n + z_max_n) / 2
            
            # 如果正常范围更合理，使用正常范围
            if max_range_n < max_range * 0.8:  # 正常范围明显小于全部范围
                print(f"使用正常关节范围: {max_range_n:.1f} cm")
                x_center, y_center, z_center = x_center_n, y_center_n, z_center_n
                max_range = max_range_n
                half_range = max_range / 2 + margin
        
        # 3D视图
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 分类关节点
        root_joints = []
        radar_joints = []
        limb_end_joints = []
        head_joints = []
        normal_joints = []
        
        # 定义关节类型
        limb_end_names = ['RightFoot', 'LeftFoot', 'RightHand', 'LeftHand', 
                         'RightToe', 'LeftToe', 'RightFingerTip', 'LeftFingerTip']
        head_names = ['Head', 'Neck', 'Neck1', 'HeadTop_End']
        root_names = ['Hips', 'Pelvis', 'pelvis', 'hips']
        
        for i, joint_name in enumerate(joint_names):
            pos = positions[i]
            
            if joint_name in root_names:
                root_joints.append((i, joint_name, pos))
            elif joint_name == 'Mid360Radar':
                radar_joints.append((i, joint_name, pos))
            elif any(end_name in joint_name for end_name in limb_end_names):
                limb_end_joints.append((i, joint_name, pos))
            elif any(head_name in joint_name for head_name in head_names):
                head_joints.append((i, joint_name, pos))
            else:
                normal_joints.append((i, joint_name, pos))
        
        # 打印关节分类信息
        print(f"\n关节分类统计:")
        print(f"  根节点 ({len(root_joints)}): {[name for _, name, _ in root_joints]}")
        print(f"  雷达 ({len(radar_joints)}): {[name for _, name, _ in radar_joints]}")
        print(f"  四肢末端 ({len(limb_end_joints)}): {[name for _, name, _ in limb_end_joints]}")
        print(f"  头部 ({len(head_joints)}): {[name for _, name, _ in head_joints]}")
        print(f"  普通关节 ({len(normal_joints)}): 共{len(normal_joints)}个")
        
        # 检查极端位置的关节
        # 检查双脚Y坐标一致性（下蹲动作中双脚应在同一水平面）
        foot_joints = {}
        for i, joint_name in enumerate(joint_names):
            if 'Foot' in joint_name and 'Hand' not in joint_name:
                foot_joints[joint_name] = (i, positions[i])
        
        if len(foot_joints) >= 2:
            foot_positions = [pos for _, pos in foot_joints.values()]
            foot_y_coords = [pos[1] for pos in foot_positions]
            y_diff = max(foot_y_coords) - min(foot_y_coords)

            if y_diff > 30:  # 双脚Y坐标差异超过30cm认为异常
                print(f"\n警告: 检测到双脚高度差异过大 ({y_diff:.1f}cm):")
                for name, (i, pos) in foot_joints.items():
                    print(f"  {name}: Y={pos[1]:.1f}cm")
                print(f"  建议检查BVH数据质量或运动学计算")
        
        extreme_joints = []
        for i, joint_name in enumerate(joint_names):
            pos = positions[i]
            if abs(pos[0]) > 50 or abs(pos[1]) > 150 or abs(pos[2]) > 200:
                extreme_joints.append((joint_name, pos))
        
        # if extreme_joints:
        #     print(f"\n极端位置关节 ({len(extreme_joints)}个):")
        #     for name, pos in extreme_joints:
        #         print(f"  {name}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        
        # 显示位置范围统计
        print(f"\n所有关节位置统计:")
        print(f"  X范围: [{np.min(positions[:, 0]):.1f}, {np.max(positions[:, 0]):.1f}] cm")
        print(f"  Y范围: [{np.min(positions[:, 1]):.1f}, {np.max(positions[:, 1]):.1f}] cm") 
        print(f"  Z范围: [{np.min(positions[:, 2]):.1f}, {np.max(positions[:, 2]):.1f}] cm")
        
        # 绘制不同类型的关节点
        if root_joints:
            root_pos = np.array([pos for _, _, pos in root_joints])
            ax1.scatter(root_pos[:, 0], root_pos[:, 1], root_pos[:, 2], 
                       c='green', s=120, marker='s', label=f'根节点 ({len(root_joints)})', 
                       edgecolors='black', alpha=0.9)
            
        if radar_joints:
            radar_pos = np.array([pos for _, _, pos in radar_joints])
            ax1.scatter(radar_pos[:, 0], radar_pos[:, 1], radar_pos[:, 2], 
                       c='red', s=150, marker='^', label=f'雷达 ({len(radar_joints)})', 
                       edgecolors='black', alpha=0.9)
            
        if limb_end_joints:
            limb_pos = np.array([pos for _, _, pos in limb_end_joints])
            ax1.scatter(limb_pos[:, 0], limb_pos[:, 1], limb_pos[:, 2], 
                       c='orange', s=80, marker='d', label=f'四肢末端 ({len(limb_end_joints)})', 
                       edgecolors='black', alpha=0.8)
            
        if head_joints:
            head_pos = np.array([pos for _, _, pos in head_joints])
            ax1.scatter(head_pos[:, 0], head_pos[:, 1], head_pos[:, 2], 
                       c='purple', s=100, marker='*', label=f'头部 ({len(head_joints)})', 
                       edgecolors='black', alpha=0.8)
            
        if normal_joints:
            normal_pos = np.array([pos for _, _, pos in normal_joints])
            ax1.scatter(normal_pos[:, 0], normal_pos[:, 1], normal_pos[:, 2], 
                       c='blue', s=30, alpha=0.6, label=f'普通关节 ({len(normal_joints)})')
        
        # # 标记异常点
        # if np.any(outlier_mask):
        #     outlier_positions = positions[outlier_mask]
        #     outlier_indices = np.where(outlier_mask)[0]
        #     ax1.scatter(outlier_positions[:, 0], outlier_positions[:, 1], outlier_positions[:, 2], 
        #                c='yellow', s=100, marker='x', label=f'异常点 ({len(outlier_indices)})', 
        #                edgecolors='red', linewidth=2, alpha=1.0)
            
        #     # 打印异常点详情
        #     print(f"\n异常关节点详情:")
        #     for idx in outlier_indices:
        #         joint_name = joint_names[idx] if idx < len(joint_names) else f"关节{idx}"
        #         pos = positions[idx]
        #         distance = distances_from_center[idx]
        #         print(f"  {joint_name}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) - 距离中心: {distance:.1f}cm")
        
        # 绘制骨骼连接
        for joint_name in joint_names:
            parent_name = joint_parents.get(joint_name)
            if parent_name and parent_name in joint_names:
                joint_idx = joint_names.index(joint_name)
                parent_idx = joint_names.index(parent_name)
                
                joint_pos = positions[joint_idx]
                parent_pos = positions[parent_idx]
                
                # 根据关节类型设置连接线颜色
                if joint_name == 'Mid360Radar':
                    line_color = 'red'
                    line_width = 2.5
                    alpha = 0.9
                elif any(end_name in joint_name for end_name in limb_end_names):
                    line_color = 'orange'
                    line_width = 1.2
                    alpha = 0.7
                else:
                    line_color = 'gray'
                    line_width = 0.8
                    alpha = 0.5
                    
                ax1.plot3D([parent_pos[0], joint_pos[0]], 
                          [parent_pos[1], joint_pos[1]], 
                          [parent_pos[2], joint_pos[2]], 
                          line_color, alpha=alpha, linewidth=line_width)
        
        # 设置统一的坐标轴范围
        ax1.set_xlim([x_center - half_range, x_center + half_range])
        ax1.set_ylim([y_center - half_range, y_center + half_range])
        ax1.set_zlim([z_center - half_range, z_center + half_range])
        
        ax1.set_xlabel('X坐标 (cm)')
        ax1.set_ylabel('Y坐标 (cm)')
        ax1.set_zlabel('Z坐标 (cm)')
        ax1.set_title(f'3D骨架结构 - 第{frame_idx}帧')
        ax1.legend()
        
        # 设置相等的比例尺
        ax1.set_box_aspect([1,1,1])
        
        # 平面视图（也统一比例尺）
        plane_configs = [
            (222, (0, 1), ('X', 'Y'), x_center, y_center, max(x_range, y_range)),
            (223, (0, 2), ('X', 'Z'), x_center, z_center, max(x_range, z_range)),
            (224, (1, 2), ('Y', 'Z'), y_center, z_center, max(y_range, z_range))
        ]
        
        for ax_idx, plane, labels, center_a, center_b, range_ab in plane_configs:
            ax = fig.add_subplot(ax_idx)
            
            # 绘制不同类型的关节点
            if root_joints:
                root_pos_2d = np.array([[pos[plane[0]], pos[plane[1]]] for _, _, pos in root_joints])
                ax.scatter(root_pos_2d[:, 0], root_pos_2d[:, 1], 
                          c='green', s=60, marker='s', edgecolors='black', alpha=0.9)
                
            if radar_joints:
                radar_pos_2d = np.array([[pos[plane[0]], pos[plane[1]]] for _, _, pos in radar_joints])
                ax.scatter(radar_pos_2d[:, 0], radar_pos_2d[:, 1], 
                          c='red', s=80, marker='^', edgecolors='black', alpha=0.9)
                
            if limb_end_joints:
                limb_pos_2d = np.array([[pos[plane[0]], pos[plane[1]]] for _, _, pos in limb_end_joints])
                ax.scatter(limb_pos_2d[:, 0], limb_pos_2d[:, 1], 
                          c='orange', s=40, marker='d', edgecolors='black', alpha=0.8)
                
            if head_joints:
                head_pos_2d = np.array([[pos[plane[0]], pos[plane[1]]] for _, _, pos in head_joints])
                ax.scatter(head_pos_2d[:, 0], head_pos_2d[:, 1], 
                          c='purple', s=50, marker='*', edgecolors='black', alpha=0.8)
                
            if normal_joints:
                normal_pos_2d = np.array([[pos[plane[0]], pos[plane[1]]] for _, _, pos in normal_joints])
                ax.scatter(normal_pos_2d[:, 0], normal_pos_2d[:, 1], 
                          c='blue', s=15, alpha=0.6)
            
            # 标记异常点
            # if np.any(outlier_mask):
            #     outlier_pos_2d = positions[outlier_mask]
            #     ax.scatter(outlier_pos_2d[:, plane[0]], outlier_pos_2d[:, plane[1]], 
            #               c='yellow', s=50, marker='x', edgecolors='red', linewidth=1.5, alpha=1.0)
            
            # 绘制骨骼连接
            for joint_name in joint_names:
                parent_name = joint_parents.get(joint_name)
                if parent_name and parent_name in joint_names:
                    joint_idx = joint_names.index(joint_name)
                    parent_idx = joint_names.index(parent_name)
                    
                    joint_pos = positions[joint_idx]
                    parent_pos = positions[parent_idx]
                    
                    # 根据关节类型设置连接线颜色
                    if joint_name == 'Mid360Radar':
                        line_color = 'red'
                        line_width = 1.5
                        alpha = 0.8
                    elif any(end_name in joint_name for end_name in limb_end_names):
                        line_color = 'orange'
                        line_width = 0.8
                        alpha = 0.6
                    else:
                        line_color = 'gray'
                        line_width = 0.5
                        alpha = 0.4
                    
                    ax.plot([parent_pos[plane[0]], joint_pos[plane[0]]], 
                           [parent_pos[plane[1]], joint_pos[plane[1]]], 
                           line_color, alpha=alpha, linewidth=line_width)
            
            # 设置统一的坐标轴范围
            half_range_ab = range_ab / 2 + margin
            ax.set_xlim([center_a - half_range_ab, center_a + half_range_ab])
            ax.set_ylim([center_b - half_range_ab, center_b + half_range_ab])
            
            ax.set_xlabel(f'{labels[0]}坐标 (cm)')
            ax.set_ylabel(f'{labels[1]}坐标 (cm)')
            ax.set_title(f'{labels[0]}{labels[1]}平面视图')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')  # 确保比例尺相等
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = 'skeleton_visualization.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"可视化图片已保存到: {plot_path}")
        
        plt.show()
    
    def process_bvh_to_pkl(self, bvh_path, pkl_path, parent_joint='Hips'):
        """完整处理流程：从BVH到PKL，添加雷达关节"""
        print("=" * 60)
        print("开始具身智能BVH到PKL转换流程")
        print("=" * 60)
        
        # 1. 解析BVH文件
        self.parse_bvh(bvh_path)
        
        # 2. 添加mid360雷达关节
        self.add_mid360_radar_joint(parent_joint)
        
        # 3. 计算关节位置
        joint_positions = self.compute_joint_positions()
        
        # 4. 创建PKL数据
        pkl_data = self.create_pkl_data(joint_positions)
        
        # 5. 保存PKL文件
        self.save_pkl(pkl_data, pkl_path)
        
        print("=" * 60)
        print("转换完成！")
        print("=" * 60)
        
        return pkl_data


def analyze_pkl_data(data):
    """分析PKL数据并输出统计信息"""
    print("\n" + "=" * 40)
    print("数据分析报告")
    print("=" * 40)
    
    print(f"总帧数: {data['num_frames']}")
    print(f"帧时间: {data['frame_time']:.6f} 秒")
    print(f"总时长: {data['num_frames'] * data['frame_time']:.2f} 秒")
    print(f"帧率: {1/data['frame_time']:.1f} FPS")
    print(f"总关节数: {len(data['joint_names'])}")
    
    if 'mid360_radar_index' in data and data['mid360_radar_index'] >= 0:
        radar_idx = data['mid360_radar_index']
        radar_positions = data['joint_positions'][:, radar_idx, :]
        
        print(f"\nMid360雷达分析:")
        print(f"  关节索引: {radar_idx}")
        print(f"  父关节: {data.get('mid360_radar_parent', 'Unknown')}")
        print(f"  设置偏移: {data.get('mid360_radar_offset', 'Unknown')}")
        print(f"  位置范围:")
        print(f"    X: [{np.min(radar_positions[:, 0]):.2f}, {np.max(radar_positions[:, 0]):.2f}] cm")
        print(f"    Y: [{np.min(radar_positions[:, 1]):.2f}, {np.max(radar_positions[:, 1]):.2f}] cm")
        print(f"    Z: [{np.min(radar_positions[:, 2]):.2f}, {np.max(radar_positions[:, 2]):.2f}] cm")
        
        # 计算运动统计
        if len(radar_positions) > 1:
            frame_time = data['frame_time']
            velocity = np.diff(radar_positions, axis=0) / frame_time
            speed = np.linalg.norm(velocity, axis=1)
            print(f"  运动统计:")
            print(f"    平均速度: {np.mean(speed):.2f} cm/s")
            print(f"    最大速度: {np.max(speed):.2f} cm/s")
            print(f"    运动距离: {np.sum(np.linalg.norm(np.diff(radar_positions, axis=0), axis=1)):.2f} cm")


def main():
    parser = argparse.ArgumentParser(description='具身智能BVH到PKL转换工具')
    parser.add_argument('--input', '-i', default='walk.bvh', 
                       help='输入BVH文件路径 (默认: walk.bvh)')
    parser.add_argument('--output', '-o', default='walk_with_radar.pkl', 
                       help='输出PKL文件路径 (默认: walk_with_radar.pkl)')
    parser.add_argument('--radar-offset', nargs=3, type=float, default=[0.003, 46.018, -0.368],
                       help='雷达相对于Hips的偏移 [x y z] (cm) (默认: 0.003 46.018 -0.368)')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化图表')
    parser.add_argument('--save-plot', action='store_true',
                       help='保存可视化图片')
    parser.add_argument('--frame', type=int, default=None,
                       help='可视化的单个帧索引')
    parser.add_argument('--frames', nargs='+', type=int, default=None,
                       help='可视化的多个帧索引')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"雷达偏移: {args.radar_offset}")
    
    # 创建处理器
    processor = EmbodiedAIProcessor(radar_offset=args.radar_offset)
    
    # 处理BVH文件
    pkl_data = processor.process_bvh_to_pkl(args.input, args.output)
    
    # 分析数据
    analyze_pkl_data(pkl_data)
    
    # 可视化
    if args.visualize:
        print("\n生成骨架可视化...")
        
        # 确定要可视化的帧
        if args.frame is not None:
            frame_indices = [args.frame]
        elif args.frames is not None:
            frame_indices = args.frames
        else:
            frame_indices = [0]  # 默认第0帧
            
        processor.visualize_skeleton(pkl_data, frame_indices=frame_indices, save_plot=args.save_plot)


if __name__ == "__main__":
    main()
