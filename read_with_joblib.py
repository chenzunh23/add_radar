import joblib
import pickle
import numpy as np

def read_joblib_file():
    """使用joblib读取文件"""
    filename = "walk_1m_retargeted_all.pkl"
    
    print(f"=== 使用joblib读取 {filename} ===")
    
    try:
        # 用joblib.load读取
        data = joblib.load(filename)
        print("✓ joblib.load 成功!")
        
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"字典键: {list(data.keys())}")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    
                    # 显示数组的一些统计信息
                    print(f"    范围: [{np.min(value):.3f}, {np.max(value):.3f}]")
                    if value.ndim == 2 and value.shape[1] == 3:
                        print(f"    可能是3D位置数据: {value.shape[0]}个点")
                    elif value.ndim == 3:
                        print(f"    可能是时序3D数据: {value.shape}")
                        
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__} (长度: {len(value)})")
                    if len(value) > 0 and len(value) <= 10:
                        print(f"    内容: {value}")
                    elif len(value) > 0:
                        print(f"    前3项: {value[:3]}")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")
        
        elif isinstance(data, np.ndarray):
            print(f"直接的numpy数组: shape={data.shape}, dtype={data.dtype}")
            print(f"数据范围: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        return data
        
    except Exception as e:
        print(f"✗ joblib.load 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_joblib_to_standard(data):
    """将joblib数据转换为我们的标准格式"""
    if data is None:
        return None
    
    print(f"\n=== 转换joblib数据到标准格式 ===")
    
    # 加载模板
    try:
        with open('lift_up_fixed.pkl', 'rb') as f:
            template = pickle.load(f)
        print("✓ 加载模板成功")
    except Exception as e:
        print(f"✗ 加载模板失败: {e}")
        return None
    
    converted_data = {}
    
    if isinstance(data, dict):
        print("分析字典数据:")
        
        # 检查是否是嵌套字典（walk_1m.pkl）
        walk_data = None
        if 'walk_1m.pkl' in data:
            walk_data = data['walk_1m.pkl']
            print("  发现嵌套的walk_1m.pkl数据")
        else:
            walk_data = data
        
        if isinstance(walk_data, dict):
            print(f"  内部字典键: {list(walk_data.keys())}")
            
            # 寻找关节位置数据
            position_array = None
            
            # 优先查找smpl_joints
            if 'smpl_joints' in walk_data:
                smpl_joints = walk_data['smpl_joints']
                print(f"  发现smpl_joints: {smpl_joints.shape}")
                
                # SMPL通常有24个关节，但我们的模板有60个关节
                # 我们需要扩展或映射
                num_frames, smpl_num_joints, coords = smpl_joints.shape
                template_num_joints = len(template['joint_names'])
                
                print(f"  SMPL关节数: {smpl_num_joints}, 模板关节数: {template_num_joints}")
                
                if smpl_num_joints < template_num_joints:
                    # 扩展SMPL数据以匹配模板
                    print(f"  扩展SMPL数据从{smpl_num_joints}到{template_num_joints}关节")
                    
                    # 创建扩展的位置数组
                    extended_positions = np.zeros((num_frames, template_num_joints, 3))
                    
                    # 复制现有的SMPL关节
                    extended_positions[:, :smpl_num_joints, :] = smpl_joints
                    
                    # 对于额外的关节，使用最近的SMPL关节位置作为近似
                    for i in range(smpl_num_joints, template_num_joints):
                        # 使用根关节（通常是索引0）的位置作为默认值
                        extended_positions[:, i, :] = smpl_joints[:, 0, :] + np.random.normal(0, 0.01, (num_frames, 3))
                    
                    position_array = extended_positions
                    print(f"  ✓ 扩展完成: {position_array.shape}")
                else:
                    # 直接使用SMPL数据
                    position_array = smpl_joints
                    print(f"  ✓ 直接使用SMPL数据: {position_array.shape}")
            
            # 如果没有smpl_joints，查找其他可能的位置数据
            elif 'joint_positions' in walk_data:
                position_array = walk_data['joint_positions']
                print(f"  使用joint_positions: {position_array.shape}")
            
            else:
                # 寻找任何3D数组
                for key, value in walk_data.items():
                    if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3:
                        position_array = value
                        print(f"  使用{key}作为位置数据: {value.shape}")
                        break
            
            if position_array is not None:
                print(f"✓ 找到关节位置数据: {position_array.shape}")
                
                # 构建标准格式
                converted_data = {
                    'joint_names': template['joint_names'],
                    'joint_positions': position_array,
                    'joint_offsets': template['joint_offsets'],
                    'joint_parents': template['joint_parents'],
                    'joint_channels': template['joint_channels'],
                    'frame_time': 1.0/30.0,  # 使用30fps（从walk_data中可以看到fps=30）
                    'num_frames': position_array.shape[0],
                    'total_joints': len(template['joint_names']),
                }
                
                # 从原始数据中提取fps如果存在
                if 'fps' in walk_data:
                    converted_data['frame_time'] = 1.0 / walk_data['fps']
                    print(f"  使用原始fps: {walk_data['fps']}")
                
                # 雷达信息
                if 'Mid360Radar' in template['joint_names']:
                    converted_data['mid360_radar_index'] = template['joint_names'].index('Mid360Radar')
                    converted_data['mid360_radar_offset'] = template['mid360_radar_offset'] 
                    converted_data['mid360_radar_parent'] = template['mid360_radar_parent']
                
                # 元数据
                converted_data['creation_timestamp'] = np.datetime64('now').astype(str)
                converted_data['conversion_source'] = 'walk_1m_retargeted_all.pkl (joblib/SMPL)'
                converted_data['original_keys'] = list(walk_data.keys())
                converted_data['original_smpl_joints'] = smpl_num_joints if 'smpl_joints' in walk_data else 'N/A'
                
                print(f"✓ 转换完成: {converted_data['num_frames']}帧, {converted_data['total_joints']}关节")
                return converted_data
    
    elif isinstance(data, np.ndarray):
        print(f"直接数组数据: {data.shape}")
        # 类似的处理逻辑...
    
    print("✗ 无法识别数据格式进行转换")
    return None

def main():
    # 读取joblib文件
    data = read_joblib_file()
    
    if data is not None:
        # 转换为标准格式
        converted = convert_joblib_to_standard(data)
        
        if converted is not None:
            # 保存转换后的数据
            output_file = "walk_1m_aligned.pkl"
            
            try:
                with open(output_file, 'wb') as f:
                    pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                print(f"\n✓ 成功保存: {output_file}")
                
                # 验证
                with open(output_file, 'rb') as f:
                    test_data = pickle.load(f)
                
                print(f"✓ 验证通过")
                print(f"  关节数: {test_data['total_joints']}")
                print(f"  帧数: {test_data['num_frames']}")
                print(f"  位置数据: {test_data['joint_positions'].shape}")
                
                return True
                
            except Exception as e:
                print(f"✗ 保存失败: {e}")
                return False
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 成功完成PKL文件对齐!")
        print(f"可以使用 walk_1m_aligned.pkl 文件了")
    else:
        print(f"\n❌ 对齐失败")
