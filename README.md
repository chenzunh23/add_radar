# 具身智能BVH到PKL转换工具套件

## 项目概述

这是一个完整的具身智能处理工具套件，提供两种不同的BVH到PKL转换方法，支持雷达关节集成、任意帧可视化和专业的数据分析功能。专门设计用于机器人运动数据处理和具身智能研究。

## 核心功能特性

### 🔄 **双转换器支持**
- ✅ **原始处理器** (`embodied_ai_processor.py`) - 传统BVH到PKL转换
- ✅ **joblib转换器** (`bvh_to_joblib_converter.py`) - 新式joblib格式，支持ZXY坐标变换

### 🎯 **雷达关节集成**
- ✅ **智能雷达定位**: Mid360雷达关节自动添加到骨架结构
- ✅ **旋转跟随**: 雷达正确跟随根关节(Hips/Pelvis)旋转，非固定偏移
- ✅ **相对定位**: 雷达位置相对于机器人骨盆精确定位

### 📊 **高级可视化**
- ✅ **任意帧查看**: 支持单帧或多帧指定查看
- ✅ **关节分类**: 根节点、雷达、四肢末端、头部、普通关节分类显示
- ✅ **完整图例**: 详细的关节统计和颜色说明
- ✅ **坐标变换**: 支持XYZ到ZXY坐标系变换

### 🔍 **数据分析**
- ✅ **运动统计**: 关节速度、位移、轨迹分析
- ✅ **异常检测**: 自动检测位置异常的关节点
- ✅ **双脚约束**: 可选的双脚高度约束功能
- ✅ **中文界面**: 完整的中文标签和字体支持

## 文件结构

```bash
retarget/
├── embodied_ai_processor.py      # 原始BVH到PKL转换器
├── bvh_to_joblib_converter.py    # 新式joblib格式转换器（支持ZXY坐标变换）
├── analyze_joblib_smpl.py        # 通用joblib SMPL数据可视化工具
├── verify_radar_rotation.py      # 雷达旋转功能验证工具
├── walk.bvh                      # 示例BVH文件
├── lift_up.bvh                   # 示例BVH文件（举手动作）
├── g1_23dof_rev_1_0.urdf        # G1机器人模型文件
├── g1_29dof_rev_1_0.urdf        # G1机器人模型文件
└── README.md                     # 本文档
```

## 安装依赖

```bash
pip install numpy matplotlib joblib pickle argparse
```

## 使用方法

### 1. 原始处理器 (embodied_ai_processor.py)

#### 基本使用
```bash
python embodied_ai_processor.py --input walk.bvh --output walk_with_radar.pkl
```

#### 带可视化和任意帧查看
```bash
# 查看单个帧
python embodied_ai_processor.py --input walk.bvh --visualize --frame 100

# 查看多个帧
python embodied_ai_processor.py --input walk.bvh --visualize --frames 0 50 100 200

# 自定义雷达位置
python embodied_ai_processor.py --input walk.bvh --radar-offset 0.003 46.018 -0.368 --visualize
```

### 2. joblib转换器 (bvh_to_joblib_converter.py)

#### 基本转换（推荐）
```bash
# 转换为joblib格式（支持ZXY坐标变换）
python bvh_to_joblib_converter.py walk.bvh --output walk_zxy_transformed.pkl

# 禁用双脚约束
python bvh_to_joblib_converter.py walk.bvh --output walk_no_constraint.pkl --no-foot-constraint

# 转换后分析
python bvh_to_joblib_converter.py walk.bvh --output walk_analyzed.pkl --analyze
```

### 3. 通用可视化工具 (analyze_joblib_smpl.py)

#### 任意帧可视化
```bash
# 查看默认关键帧
python analyze_joblib_smpl.py walk_zxy_transformed.pkl

# 查看指定单帧
python analyze_joblib_smpl.py walk_zxy_transformed.pkl --frame 150

# 查看多个指定帧
python analyze_joblib_smpl.py walk_zxy_transformed.pkl --frames 0 100 200 300 400
```

### 4. 雷达功能验证
```bash
# 验证雷达旋转跟随功能
python verify_radar_rotation.py
```

## 参数说明

### embodied_ai_processor.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input`, `-i` | 输入BVH文件路径 | `walk.bvh` |
| `--output`, `-o` | 输出PKL文件路径 | `walk_with_radar.pkl` |
| `--radar-offset` | 雷达相对于Hips的偏移 [x y z] (cm) | `[0.003, 46.018, -0.368]` |
| `--visualize` | 生成可视化图表 | `False` |
| `--save-plot` | 保存可视化图片 | `False` |
| `--frame` | 可视化的单个帧索引 | `None` |
| `--frames` | 可视化的多个帧索引 | `None` |

### bvh_to_joblib_converter.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入BVH文件路径（位置参数） | 必需 |
| `--output`, `-o` | 输出joblib PKL文件路径 | `{输入文件名}_joblib.pkl` |
| `--no-foot-constraint` | 禁用双脚高度约束 | `False` |
| `--analyze` | 转换后分析输出文件 | `False` |

### analyze_joblib_smpl.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `filename` | 要分析的PKL文件路径（位置参数） | `lift_up_joblib.pkl` |
| `--frame`, `-f` | 指定要可视化的帧索引 | `None` |
| `--frames`, `-fs` | 指定多个要可视化的帧索引 | `None` |

## 雷达关节配置

### 坐标系说明

**原始XYZ坐标系 → ZXY坐标系变换**：

- 新X = 原Z（前后方向 → 上下方向）
- 新Y = 原X（左右方向 → 左右方向）  
- 新Z = 原Y（上下方向 → 前后方向）

### 雷达偏移配置

Mid360雷达关节相对于机器人Hips（骨盆）关节的偏移：

**原始XYZ坐标系**: `[46.018, -0.368, 0.003]` cm

- X轴: 46.018cm（向上46cm）
- Y轴: -0.368cm（向左0.368cm）
- Z轴: 0.003cm（向前0.003cm）

**转换后ZXY坐标系**: `[-0.368, 0.003, 46.018]` cm

- X轴: -0.368cm（向左0.368cm）
- Y轴: 0.003cm（几乎无偏移）
- Z轴: 46.018cm（向上46cm）

### 雷达旋转跟随

✅ **智能旋转**: 雷达会正确跟随Hips/Pelvis的旋转而移动，不再是固定偏移
✅ **相对定位**: 保持与根关节的相对位置关系
✅ **真实轨迹**: 生成符合物理规律的雷达运动轨迹

## 数据格式说明

### 原始PKL格式 (embodied_ai_processor.py)

```python
{
    'joint_names': list,              # 关节名称列表（60个关节，包含雷达）
    'joint_positions': array,         # 关节位置 [frames, joints, 3]
    'joint_offsets': dict,            # 关节偏移字典
    'joint_parents': dict,            # 关节父子关系
    'joint_channels': dict,           # 关节通道信息
    'frame_time': float,              # 帧时间间隔
    'num_frames': int,                # 总帧数
    'mid360_radar_index': int,        # 雷达关节索引
    'mid360_radar_offset': array,     # 雷达偏移向量
    'mid360_radar_parent': str,       # 雷达父关节名称
    'creation_timestamp': str,        # 创建时间戳
    'total_joints': int,              # 总关节数
    'original_motion_data_shape': tuple  # 原始运动数据形状
}
```

### joblib SMPL格式 (bvh_to_joblib_converter.py)

```python
{
    'filename.pkl': {                 # 外层包装
        'root_trans_offset': array,   # 根部位移 [frames, 3]
        'pose_aa': array,             # 姿态轴角 [frames, 35, 3] 
        'dof': array,                 # 自由度参数 [frames, 29]
        'root_rot': array,            # 根部旋转四元数 [frames, 4]
        'smpl_joints': array,         # SMPL关节 [frames, 46, 3] (45+雷达)
        'hand_dof': array,            # 手部自由度 [frames, 12]
        'fps': int,                   # 帧率
        'fit_loss': float             # 拟合损失
    }
}
```

## 可视化功能详解

### 关节分类可视化

| 类型 | 颜色 | 标记 | 描述 |
|------|------|------|------|
| **根节点** | 绿色 | 方块 | Pelvis/Hips根关节 |
| **雷达** | 红色 | 三角 | Mid360Radar雷达关节 |
| **四肢末端** | 橙色 | 菱形 | 手足等末端关节 |
| **头部** | 紫色 | 星形 | 头部和颈部关节 |
| **普通关节** | 蓝色 | 圆点 | 其他身体关节 |

### 多视角显示

1. **3D骨架结构**: 完整的三维骨架显示
2. **XY平面视图**: 俯视图（从上往下看）
3. **XZ平面视图**: 侧视图（从侧面看）
4. **YZ平面视图**: 正视图（从正面看）

### 任意帧查看

- **默认模式**: 显示开始、中间、结束三个关键帧
- **单帧模式**: `--frame N` 查看指定帧
- **多帧模式**: `--frames 0 50 100 200` 查看多个指定帧
- **帧验证**: 自动检查帧索引有效性，无效帧自动跳过

## 数据分析功能

### 运动统计分析

- **双脚高度差异**: 分析步态中双脚的高度变化
- **骨盆移动**: 计算根关节的平均速度和最大速度
- **雷达轨迹**: 雷达关节的运动路径和速度统计
- **关节范围**: 各关节的位置范围和变化幅度

### 异常检测

- 自动检测距离中心超过3倍标准差的异常关节
- 双脚高度差异超过30cm的警告
- 数据质量评估和建议

## 技术特性

### 坐标变换

- **XYZ → ZXY变换**: 适配不同坐标系需求
- **单位转换**: 厘米到米的自动转换
- **旋转处理**: 正确的欧拉角旋转计算

### 性能优化

- **批量处理**: 支持多帧并行计算
- **内存管理**: 优化大型BVH文件的内存使用
- **缓存机制**: 重复计算的结果缓存

### 错误处理

- **文件验证**: 自动检查输入文件格式和完整性
- **参数校验**: 验证所有输入参数的有效性
- **异常恢复**: 详细的错误信息和恢复建议

## 示例输出

### 数据分析报告示例

```text
=== 分析文件: walk_zxy_transformed.pkl ===
✓ 加载成功
数据类型: <class 'dict'>
顶层键: ['walk.pkl']

=== SMPL运动分析 ===
帧数: 498
关节数: 46
坐标维度: 3
数据范围:
  X: [-0.007, 2.727]
  Y: [-0.613, 0.269]
  Z: [0.089, 1.683]
  FPS: 90
  时长: 5.53秒

=== 关键关节分析 ===
双脚高度差异:
  最大差异: 0.462m
  平均差异: 0.101m
骨盆移动:
  平均速度: 0.0045m/帧
  最大速度: 0.0133m/帧
```

### 雷达旋转验证示例

```text
=== 分析 修改后（跟随旋转） ===
偏移变化标准差: X=0.0042, Y=0.0250, Z=0.0248
雷达运动:
  平均速度: 0.0048 m/帧
  最大速度: 0.0162 m/帧
  总移动距离: 2.365 m

结论: 雷达位置有显著差异（跟随旋转功能正常）
```

## 最佳实践

### 推荐工作流程

1. **使用joblib转换器**进行初始转换:

   ```bash
   python bvh_to_joblib_converter.py input.bvh --output result.pkl
   ```

2. **验证雷达功能**:

   ```bash
   python verify_radar_rotation.py
   ```

3. **任意帧可视化分析**:

   ```bash
   python analyze_joblib_smpl.py result.pkl --frames 0 100 200 300
   ```

### 常见问题解决

1. **雷达位置不正确**: 检查雷达偏移参数是否适合您的机器人配置
2. **可视化显示异常**: 确保安装了正确的中文字体
3. **转换失败**: 验证BVH文件格式和Hips关节存在

### 性能建议

- 大型BVH文件（>1000帧）建议使用joblib转换器
- 可视化时避免同时显示过多帧（建议<10帧）
- 使用`--no-foot-constraint`可以提高转换速度

## 更新日志

### v2.0 (最新)

- ✅ 新增joblib转换器，支持ZXY坐标变换
- ✅ 实现雷达旋转跟随功能  
- ✅ 添加任意帧查看功能
- ✅ 完善关节分类和图例系统
- ✅ 新增雷达功能验证工具

### v1.0 (原始版本)

- ✅ 基础BVH到PKL转换
- ✅ 简单雷达关节添加
- ✅ 基础3D可视化功能
