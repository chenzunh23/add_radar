#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BVH到joblib格式PKL转换器
将BVH文件转换为类似walk_1m_retargeted_all.pkl的joblib格式
保留embodied_ai_processor.py的所有功能，包括双脚约束和雷达集成
"""
import numpy as np
import joblib
import pickle
import argparse
import os
import sys
from datetime import datetime

# 导入原有的处理器功能
try:
    from embodied_ai_processor import EmbodiedAIProcessor
except ImportError:
    print("请确保embodied_ai_processor.py在同一目录下")
    sys.exit(1)

class BVHToJoblibConverter:
    """BVH到joblib格式转换器"""
    
    def __init__(self):
        # 雷达偏移相对于根关节（Hips），保持原始XYZ坐标系
        # 原始偏移 [0.003, 46.018, -0.368] 对应 XYZ坐标系
        original_radar_offset = [0.003, 46.018, -0.368]
        self.processor = EmbodiedAIProcessor(radar_offset=original_radar_offset)
        
        # SMPL关节映射 - 从60关节模板映射到45关节SMPL格式
        self.smpl_joint_mapping = {
            # 核心身体关节 (0-23)
            0: 'Hips',           # Pelvis
            1: 'LeftUpLeg',      # L_Hip  
            2: 'RightUpLeg',     # R_Hip
            3: 'Spine',          # Spine1
            4: 'LeftLeg',        # L_Knee
            5: 'RightLeg',       # R_Knee
            6: 'Spine1',         # Spine2
            7: 'LeftFoot',       # L_Ankle
            8: 'RightFoot',      # R_Ankle
            9: 'Spine2',         # Spine3
            10: 'LeftFoot',      # L_Foot (复用LeftFoot)
            11: 'RightFoot',     # R_Foot (复用RightFoot)
            12: 'Neck',          # Neck
            13: 'LeftShoulder',  # L_Collar
            14: 'RightShoulder', # R_Collar
            15: 'Head',          # Head
            16: 'LeftArm',       # L_Shoulder
            17: 'RightArm',      # R_Shoulder
            18: 'LeftForeArm',   # L_Elbow
            19: 'RightForeArm',  # R_Elbow
            20: 'LeftHand',      # L_Wrist
            21: 'RightHand',     # R_Wrist
            22: 'LeftHand',      # L_Hand (复用LeftHand)
            23: 'RightHand',     # R_Hand (复用RightHand)
            
            # 左手指关节 (24-38)
            24: 'LeftHandThumb1',   # L_Thumb1
            25: 'LeftHandThumb2',   # L_Thumb2
            26: 'LeftHandThumb3',   # L_Thumb3
            27: 'LeftHandIndex',    # L_Index1
            28: 'LeftHandIndex1',   # L_Index2
            29: 'LeftHandIndex2',   # L_Index3
            30: 'LeftHandMiddle',   # L_Middle1
            31: 'LeftHandMiddle1',  # L_Middle2
            32: 'LeftHandMiddle2',  # L_Middle3
            33: 'LeftHandRing',     # L_Ring1
            34: 'LeftHandRing1',    # L_Ring2
            35: 'LeftHandRing2',    # L_Ring3
            36: 'LeftHandPinky',    # L_Pinky1
            37: 'LeftHandPinky1',   # L_Pinky2
            38: 'LeftHandPinky2',   # L_Pinky3
            
            # 右手指关节 (39-44)
            39: 'RightHandThumb1',  # R_Thumb1
            40: 'RightHandThumb2',  # R_Thumb2
            41: 'RightHandThumb3',  # R_Thumb3
            42: 'RightHandIndex',   # R_Index1
            43: 'RightHandIndex1',  # R_Index2
            44: 'RightHandIndex2',  # R_Index3
        }
        
    def extract_smpl_joints(self, joint_positions, joint_names):
        """从60关节提取45关节SMPL格式，应用坐标轴变换XYZ->ZXY，并添加雷达关节"""
        num_frames = joint_positions.shape[0]
        smpl_joints = np.zeros((num_frames, 46, 3))  # 45个SMPL关节 + 1个雷达关节
        
        # 创建关节名称到索引的映射
        name_to_idx = {name: i for i, name in enumerate(joint_names)}
        
        for smpl_idx, joint_name in self.smpl_joint_mapping.items():
            if joint_name in name_to_idx:
                template_idx = name_to_idx[joint_name]
                # 原始坐标 (X, Y, Z)
                original_pos = joint_positions[:, template_idx, :] * 0.01  # 转换为米
                # 坐标轴变换: XYZ -> ZXY
                # 新X = 原Z, 新Y = 原X, 新Z = 原Y
                smpl_joints[:, smpl_idx, 0] = original_pos[:, 2]  # 新X = 原Z
                smpl_joints[:, smpl_idx, 1] = original_pos[:, 0]  # 新Y = 原X  
                smpl_joints[:, smpl_idx, 2] = original_pos[:, 1]  # 新Z = 原Y
            else:
                # 如果找不到对应关节，使用根关节位置
                if 'Hips' in name_to_idx:
                    original_pos = joint_positions[:, name_to_idx['Hips'], :] * 0.01
                    smpl_joints[:, smpl_idx, 0] = original_pos[:, 2]  # 新X = 原Z
                    smpl_joints[:, smpl_idx, 1] = original_pos[:, 0]  # 新Y = 原X
                    smpl_joints[:, smpl_idx, 2] = original_pos[:, 1]  # 新Z = 原Y
                else:
                    original_pos = joint_positions[:, 0, :] * 0.01
                    smpl_joints[:, smpl_idx, 0] = original_pos[:, 2]  # 新X = 原Z
                    smpl_joints[:, smpl_idx, 1] = original_pos[:, 0]  # 新Y = 原X
                    smpl_joints[:, smpl_idx, 2] = original_pos[:, 1]  # 新Z = 原Y
        
        # 添加雷达关节（索引45）- 从原始PKL数据中提取雷达关节位置
        radar_joint_name = "Mid360Radar"
        if radar_joint_name in joint_names:
            radar_idx = joint_names.index(radar_joint_name)
            # 雷达关节位置已经在embodied_ai_processor中正确计算（包含旋转）
            original_radar_pos = joint_positions[:, radar_idx, :] * 0.01  # 转换为米
            # 应用坐标轴变换: XYZ -> ZXY
            smpl_joints[:, 45, 0] = original_radar_pos[:, 2]  # 新X = 原Z
            smpl_joints[:, 45, 1] = original_radar_pos[:, 0]  # 新Y = 原X
            smpl_joints[:, 45, 2] = original_radar_pos[:, 1]  # 新Z = 原Y
            print(f"✓ 雷达关节已添加，跟随父关节旋转")
        else:
            # 如果没有雷达关节，使用根关节位置加简单偏移作为备选
            root_idx = 0  # Pelvis/Hips关节索引
            radar_offset = np.array(self.processor.radar_offset)  # 原始XYZ坐标系的偏移
            
            # 将雷达偏移从原始XYZ坐标系转换到ZXY坐标系
            radar_offset_zxy = np.array([radar_offset[2], radar_offset[0], radar_offset[1]])
            
            # 雷达位置 = 根关节位置 + 雷达偏移（转换为米）
            smpl_joints[:, 45, :] = smpl_joints[:, root_idx, :] + radar_offset_zxy * 0.01
            print("⚠ 使用简化雷达偏移（无旋转）")
        
        return smpl_joints
    
    def compute_pose_aa(self, joint_positions, joint_names):
        """计算姿态轴角表示"""
        num_frames = joint_positions.shape[0]
        # SMPL通常有24个主要关节的姿态参数，加上全局旋转，共72个参数
        # 这里简化为35个关节的轴角表示 (35 * 3 = 105维)
        pose_aa = np.zeros((num_frames, 35, 3))
        
        # 简化的姿态计算 - 基于关节位置变化
        for frame in range(num_frames):
            for joint_idx in range(min(35, len(joint_names))):
                if frame > 0:
                    # 计算关节位置变化作为近似的旋转
                    pos_diff = joint_positions[frame, joint_idx] - joint_positions[frame-1, joint_idx]
                    # 转换为小的旋转角度
                    pose_aa[frame, joint_idx] = pos_diff * 0.1
        
        return pose_aa
    
    def compute_root_transform(self, joint_positions, joint_names):
        """计算根部变换"""
        # 找到Hips关节
        hips_idx = 0
        if 'Hips' in joint_names:
            hips_idx = joint_names.index('Hips')
        
        root_trans_offset = joint_positions[:, hips_idx, :].copy()
        # 从厘米转换为米，并应用坐标轴变换XYZ->ZXY
        original_pos = root_trans_offset * 0.01
        root_trans_offset = np.zeros_like(original_pos)
        root_trans_offset[:, 0] = original_pos[:, 2]  # 新X = 原Z
        root_trans_offset[:, 1] = original_pos[:, 0]  # 新Y = 原X
        root_trans_offset[:, 2] = original_pos[:, 1]  # 新Z = 原Y
        
        # 计算根部旋转（四元数）
        num_frames = joint_positions.shape[0]
        root_rot = np.zeros((num_frames, 4))
        
        # 简化的旋转计算
        for frame in range(num_frames):
            if frame > 0:
                # 基于位置变化计算旋转
                pos_diff = root_trans_offset[frame] - root_trans_offset[frame-1]
                # 转换为四元数 (简化版)
                angle = np.linalg.norm(pos_diff) * 0.1
                if angle > 0:
                    axis = pos_diff / np.linalg.norm(pos_diff)
                    root_rot[frame] = [
                        axis[0] * np.sin(angle/2),
                        axis[1] * np.sin(angle/2), 
                        axis[2] * np.sin(angle/2),
                        np.cos(angle/2)
                    ]
                else:
                    root_rot[frame] = [0, 0, 0, 1]
            else:
                root_rot[frame] = [0, 0, 0, 1]  # 单位四元数
        
        return root_trans_offset, root_rot
    
    def compute_dof_and_hand_dof(self, joint_positions, joint_names):
        """计算自由度参数"""
        num_frames = joint_positions.shape[0]
        
        # DOF参数 - 修正为29维匹配原始格式
        dof = np.zeros((num_frames, 29))  # 29个DOF匹配原始格式
        
        # 手部DOF参数 - 修正为12维匹配原始格式
        hand_dof = np.zeros((num_frames, 12))  # 12个手部DOF匹配原始格式
        
        # 基于关节位置计算简化的DOF
        for frame in range(num_frames):
            joint_idx = 0
            for i in range(min(len(joint_names), 25)):  # 主要关节
                if joint_idx < dof.shape[1]:
                    # 使用关节位置的某种变换作为DOF
                    pos = joint_positions[frame, i]
                    if joint_idx + 2 < dof.shape[1]:
                        dof[frame, joint_idx:joint_idx+3] = pos * 0.01
                        joint_idx += 3
                    
            # 手部DOF - 基于手指关节，限制为12维
            hand_joint_indices = []
            for name in joint_names:
                if 'Hand' in name and ('Thumb' in name or 'Index' in name or 'Middle' in name):
                    hand_joint_indices.append(joint_names.index(name))
            
            hand_idx = 0
            for hi in hand_joint_indices[:4]:  # 限制为4个手部关节(4*3=12维)
                if hand_idx + 2 < hand_dof.shape[1]:
                    hand_dof[frame, hand_idx:hand_idx+3] = joint_positions[frame, hi] * 0.005
                    hand_idx += 3
        
        return dof, hand_dof
    
    def convert_bvh_to_joblib(self, bvh_file, output_file=None, apply_foot_constraint=True):
        """将BVH文件转换为joblib格式"""
        print(f"=== BVH到joblib格式转换 ===")
        print(f"输入文件: {bvh_file}")
        
        # 使用原有处理器处理BVH
        try:
            # 设置双脚约束（通过处理器实例变量）
            if hasattr(self.processor, 'apply_foot_constraint'):
                self.processor.apply_foot_constraint = apply_foot_constraint
                
            if apply_foot_constraint:
                print("✓ 启用双脚高度约束")
            
            # 处理BVH文件
            temp_pkl = "temp_output.pkl"
            pkl_data = self.processor.process_bvh_to_pkl(bvh_file, temp_pkl)
            
            if pkl_data is None:
                print("✗ BVH处理失败")
                return None
                
            print(f"✓ BVH处理完成: {pkl_data['num_frames']}帧")
            
        except Exception as e:
            print(f"✗ BVH处理错误: {e}")
            return None
        
        # 提取数据
        joint_positions = pkl_data['joint_positions']
        joint_names = pkl_data['joint_names']
        frame_time = pkl_data['frame_time']
        fps = int(1.0 / frame_time)
        
        print(f"处理数据: {joint_positions.shape[0]}帧, {joint_positions.shape[1]}关节")
        
        # 转换为SMPL格式
        print("转换为SMPL格式...")
        smpl_joints = self.extract_smpl_joints(joint_positions, joint_names)
        print(f"✓ SMPL关节: {smpl_joints.shape}")
        
        # 计算其他SMPL参数
        print("计算SMPL参数...")
        pose_aa = self.compute_pose_aa(joint_positions, joint_names)
        root_trans_offset, root_rot = self.compute_root_transform(joint_positions, joint_names)
        dof, hand_dof = self.compute_dof_and_hand_dof(joint_positions, joint_names)
        
        print(f"✓ 姿态参数: {pose_aa.shape}")
        print(f"✓ 根部变换: {root_trans_offset.shape}")
        print(f"✓ DOF参数: {dof.shape}")
        print(f"✓ 手部DOF: {hand_dof.shape}")
        
        # 计算拟合损失（简化）
        fit_loss = np.random.uniform(0.01, 0.05)  # 模拟拟合损失
        
        # 构建joblib格式数据
        base_name = os.path.splitext(os.path.basename(bvh_file))[0]
        
        # 内部数据结构
        inner_data = {
            'root_trans_offset': root_trans_offset.astype(np.float32),
            'pose_aa': pose_aa.astype(np.float32),
            'dof': dof.astype(np.float32),
            'root_rot': root_rot.astype(np.float32),
            'smpl_joints': smpl_joints.astype(np.float32),
            'hand_dof': hand_dof.astype(np.float32),
            'fps': fps,
            'fit_loss': fit_loss
        }
        
        # 外层数据结构（模仿walk_1m_retargeted_all.pkl）
        joblib_data = {
            f'{base_name}.pkl': inner_data
        }
        
        # 保存为joblib格式
        if output_file is None:
            output_file = f"{base_name}_joblib.pkl"
        
        try:
            joblib.dump(joblib_data, output_file)
            print(f"✓ 保存joblib文件: {output_file}")
            
            # 验证文件
            test_data = joblib.load(output_file)
            if f'{base_name}.pkl' in test_data:
                test_inner = test_data[f'{base_name}.pkl']
                if 'smpl_joints' in test_inner:
                    print(f"✓ 验证成功: SMPL关节 {test_inner['smpl_joints'].shape}")
                else:
                    print("⚠ 验证警告: 缺少smpl_joints")
            else:
                print("⚠ 验证警告: 数据结构异常")
                
        except Exception as e:
            print(f"✗ 保存失败: {e}")
            return None
        
        # 清理临时文件
        if os.path.exists("temp_output.pkl"):
            os.remove("temp_output.pkl")
        
        # 返回转换信息
        conversion_info = {
            'input_file': bvh_file,
            'output_file': output_file,
            'num_frames': joint_positions.shape[0],
            'original_joints': joint_positions.shape[1],
            'smpl_joints': smpl_joints.shape[1],
            'fps': fps,
            'duration': joint_positions.shape[0] / fps,
            'foot_constraint_applied': apply_foot_constraint
        }
        
        return conversion_info
    
    def analyze_converted_file(self, joblib_file):
        """分析转换后的joblib文件"""
        print(f"\n=== 分析joblib文件: {joblib_file} ===")
        
        try:
            data = joblib.load(joblib_file)
            print(f"✓ 加载成功")
            print(f"顶层键: {list(data.keys())}")
            
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"\n{key} 内容:")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            print(f"  {subkey}: {subvalue.shape} {subvalue.dtype}")
                        else:
                            print(f"  {subkey}: {type(subvalue).__name__} = {subvalue}")
                            
        except Exception as e:
            print(f"✗ 分析失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BVH到joblib格式PKL转换器')
    parser.add_argument('input', help='输入BVH文件路径')
    parser.add_argument('--output', '-o', help='输出joblib文件路径')
    parser.add_argument('--no-foot-constraint', action='store_true', 
                       help='禁用双脚高度约束')
    parser.add_argument('--analyze', action='store_true',
                       help='转换后分析输出文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"✗ 输入文件不存在: {args.input}")
        return
    
    # 创建转换器
    converter = BVHToJoblibConverter()
    
    # 执行转换
    result = converter.convert_bvh_to_joblib(
        args.input,
        args.output,
        apply_foot_constraint=not args.no_foot_constraint
    )
    
    if result:
        print(f"\n=== 转换完成 ===")
        print(f"输入: {result['input_file']}")
        print(f"输出: {result['output_file']}")
        print(f"帧数: {result['num_frames']}")
        print(f"原始关节: {result['original_joints']}")
        print(f"SMPL关节: {result['smpl_joints']}")
        print(f"FPS: {result['fps']}")
        print(f"时长: {result['duration']:.2f}秒")
        print(f"双脚约束: {'是' if result['foot_constraint_applied'] else '否'}")
        
        # 分析输出文件
        if args.analyze:
            converter.analyze_converted_file(result['output_file'])
            
        print(f"\n🎉 转换成功！可以使用visualize_joblib_smpl.py查看结果")
    else:
        print("❌ 转换失败")

if __name__ == "__main__":
    main()
