# Compute the offset of the mid360 from a parent joint
# The parameters of the parent are found in g1_29dof_rev_1_0.urdf
import numpy as np
import re
import xml.etree.ElementTree as ET

def parse_urdf(urdf_file):
    """
    Parse a URDF file to extract joint and link offset information.
    
    Args:
        urdf_file (str): Path to the URDF file
        
    Returns:
        dict: Dictionary containing joint and link information with offsets
              Structure: {
                  'joints': {
                      'joint_name': {
                          'type': 'revolute/fixed/floating',
                          'parent': 'parent_link_name',
                          'child': 'child_link_name',
                          'origin': {'xyz': [x, y, z], 'rpy': [r, p, y]},
                          'axis': [x, y, z] (if applicable),
                          'limits': {'lower': val, 'upper': val, 'effort': val, 'velocity': val} (if applicable)
                      }
                  },
                  'links': {
                      'link_name': {
                          'inertial_origin': {'xyz': [x, y, z], 'rpy': [r, p, y]},
                          'mass': value,
                          'inertia': {'ixx': val, 'ixy': val, ...}
                      }
                  }
              }
    """
    try:
        # Parse the XML file
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        
        result = {
            'joints': {},
            'links': {}
        }
        
        # Parse joints
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            
            joint_info = {
                'type': joint_type,
                'parent': None,
                'child': None,
                'origin': {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]},
                'axis': None,
                'limits': None
            }
            
            # Get parent and child links
            parent_elem = joint.find('parent')
            if parent_elem is not None:
                joint_info['parent'] = parent_elem.get('link')
                
            child_elem = joint.find('child')
            if child_elem is not None:
                joint_info['child'] = child_elem.get('link')
            
            # Get origin (offset from parent)
            origin_elem = joint.find('origin')
            if origin_elem is not None:
                xyz_str = origin_elem.get('xyz', '0 0 0')
                rpy_str = origin_elem.get('rpy', '0 0 0')
                
                joint_info['origin']['xyz'] = [float(x) for x in xyz_str.split()]
                joint_info['origin']['rpy'] = [float(x) for x in rpy_str.split()]
            
            # Get axis (for revolute joints)
            axis_elem = joint.find('axis')
            if axis_elem is not None:
                axis_str = axis_elem.get('xyz', '0 0 0')
                joint_info['axis'] = [float(x) for x in axis_str.split()]
            
            # Get limits (for revolute joints)
            limit_elem = joint.find('limit')
            if limit_elem is not None:
                joint_info['limits'] = {
                    'lower': float(limit_elem.get('lower', '0')),
                    'upper': float(limit_elem.get('upper', '0')),
                    'effort': float(limit_elem.get('effort', '0')),
                    'velocity': float(limit_elem.get('velocity', '0'))
                }
            
            result['joints'][joint_name] = joint_info
        
        # Parse links
        for link in root.findall('link'):
            link_name = link.get('name')
            
            link_info = {
                'inertial_origin': {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]},
                'mass': 0.0,
                'inertia': {}
            }
            
            # Get inertial properties
            inertial_elem = link.find('inertial')
            if inertial_elem is not None:
                # Get inertial origin
                origin_elem = inertial_elem.find('origin')
                if origin_elem is not None:
                    xyz_str = origin_elem.get('xyz', '0 0 0')
                    rpy_str = origin_elem.get('rpy', '0 0 0')
                    
                    link_info['inertial_origin']['xyz'] = [float(x) for x in xyz_str.split()]
                    link_info['inertial_origin']['rpy'] = [float(x) for x in rpy_str.split()]
                
                # Get mass
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    link_info['mass'] = float(mass_elem.get('value', '0'))
                
                # Get inertia matrix
                inertia_elem = inertial_elem.find('inertia')
                if inertia_elem is not None:
                    inertia_attrs = ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']
                    for attr in inertia_attrs:
                        value = inertia_elem.get(attr, '0')
                        link_info['inertia'][attr] = float(value)
            
            result['links'][link_name] = link_info
        
        return result
        
    except ET.ParseError as e:
        print(f"Error parsing URDF file: {e}")
        return None
    except FileNotFoundError:
        print(f"URDF file not found: {urdf_file}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def get_joint_offset(urdf_data, joint_name):
    """
    Get the offset (translation) of a specific joint from its parent.
    
    Args:
        urdf_data (dict): Parsed URDF data from parse_urdf()
        joint_name (str): Name of the joint
        
    Returns:
        numpy.ndarray: 3D translation vector [x, y, z] or None if not found
    """
    if urdf_data is None or joint_name not in urdf_data['joints']:
        return None
    
    joint = urdf_data['joints'][joint_name]
    return np.array(joint['origin']['xyz'])

def get_link_inertial_offset(urdf_data, link_name):
    """
    Get the inertial offset (center of mass) of a specific link.
    
    Args:
        urdf_data (dict): Parsed URDF data from parse_urdf()
        link_name (str): Name of the link
        
    Returns:
        numpy.ndarray: 3D translation vector [x, y, z] or None if not found
    """
    if urdf_data is None or link_name not in urdf_data['links']:
        return None
    
    link = urdf_data['links'][link_name]
    return np.array(link['inertial_origin']['xyz'])

def compute_chain_offset(urdf_data, target_joint, base_joint='pelvis'):
    """
    Compute the cumulative offset from a base joint to a target joint
    by traversing the kinematic chain.
    
    Args:
        urdf_data (dict): Parsed URDF data from parse_urdf()
        target_joint (str): Name of the target joint
        base_joint (str): Name of the base joint/link (default: 'pelvis')
        
    Returns:
        numpy.ndarray: 3D cumulative translation vector [x, y, z] or None if path not found
    """
    if urdf_data is None:
        return None
    
    # Build a graph of parent-child relationships
    parent_to_child = {}
    child_to_parent = {}
    
    for joint_name, joint_info in urdf_data['joints'].items():
        parent = joint_info['parent']
        child = joint_info['child']
        
        if parent and child:
            if parent not in parent_to_child:
                parent_to_child[parent] = []
            parent_to_child[parent].append((child, joint_name))
            child_to_parent[child] = (parent, joint_name)
    
    # Find path from base to target
    def find_path(start, end):
        if start == end:
            return []
        
        visited = set()
        queue = [(start, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            # Check children
            if current in parent_to_child:
                for child, joint_name in parent_to_child[current]:
                    new_path = path + [(joint_name, 'forward')]
                    if child == end:
                        return new_path
                    queue.append((child, new_path))
            
            # Check parent
            if current in child_to_parent:
                parent, joint_name = child_to_parent[current]
                new_path = path + [(joint_name, 'backward')]
                if parent == end:
                    return new_path
                queue.append((parent, new_path))
        
        return None
    
    # Find the target joint's child link
    target_child = None
    for joint_name, joint_info in urdf_data['joints'].items():
        if joint_name == target_joint:
            target_child = joint_info['child']
            break
    
    if target_child is None:
        return None
    
    # Find path from base to target child link
    path = find_path(base_joint, target_child)
    if path is None:
        return None
    
    # Compute cumulative offset
    cumulative_offset = np.array([0.0, 0.0, 0.0])
    
    for joint_name, direction in path:
        joint_offset = get_joint_offset(urdf_data, joint_name)
        if joint_offset is not None:
            if direction == 'forward':
                cumulative_offset += joint_offset
            else:  # backward
                cumulative_offset -= joint_offset
    
    return cumulative_offset

def get_all_offsets_from_base(urdf_data, base_joint='pelvis'):
    """
    Compute offsets of all joints/links relative to a base joint/link.
    
    Args:
        urdf_data (dict): Parsed URDF data from parse_urdf()
        base_joint (str): Name of the base joint/link (default: 'pelvis')
        
    Returns:
        dict: Dictionary with joint names as keys and their offsets as values
    """
    if urdf_data is None:
        return {}
    
    offsets = {}
    
    # Get all joint names
    for joint_name in urdf_data['joints'].keys():
        offset = compute_chain_offset(urdf_data, joint_name, base_joint)
        if offset is not None:
            offsets[joint_name] = offset
    
    return offsets

def find_joints_by_type(urdf_data, joint_type):
    """
    Find all joints of a specific type.
    
    Args:
        urdf_data (dict): Parsed URDF data from parse_urdf()
        joint_type (str): Type of joint ('revolute', 'fixed', 'floating', etc.)
        
    Returns:
        list: List of joint names matching the specified type
    """
    if urdf_data is None:
        return []
    
    matching_joints = []
    for joint_name, joint_info in urdf_data['joints'].items():
        if joint_info['type'] == joint_type:
            matching_joints.append(joint_name)
    
    return matching_joints

def compute_relative_offset(joint1, joint2):
    """
    Compute the relative offset between two joints using urdf file g1_29dof_rev_1_0.urdf.
    
    Args:
        joint1 (string): Name of the first joint
        joint2 (string): Name of the second joint
    """
    urdf_data = parse_urdf("g1_29dof_rev_1_0.urdf")
    if urdf_data is None:
        print("URDF data not available")
        return None
    offset1 = get_joint_offset(urdf_data, joint1)
    offset2 = get_joint_offset(urdf_data, joint2)
    if offset1 is None or offset2 is None:
        print(f"Offsets for joints '{joint1}' or '{joint2}' not found")
        return None
    relative_offset = offset2 - offset1
    return relative_offset

def print_urdf_summary(urdf_data):
    """
    Print a summary of the URDF structure.
    
    Args:
        urdf_data (dict): Parsed URDF data from parse_urdf()
    """
    if urdf_data is None:
        print("No URDF data available")
        return
    
    print("URDF Summary:")
    print(f"  Total joints: {len(urdf_data['joints'])}")
    print(f"  Total links: {len(urdf_data['links'])}")
    
    # Count joints by type
    joint_types = {}
    for joint_info in urdf_data['joints'].values():
        joint_type = joint_info['type']
        joint_types[joint_type] = joint_types.get(joint_type, 0) + 1
    
    print("  Joint types:")
    for joint_type, count in joint_types.items():
        print(f"    {joint_type}: {count}")
    
    # Show root links (links that are not children of any joint)
    child_links = set()
    for joint_info in urdf_data['joints'].values():
        if joint_info['child']:
            child_links.add(joint_info['child'])
    
    root_links = []
    for link_name in urdf_data['links'].keys():
        if link_name not in child_links:
            root_links.append(link_name)
    
    print(f"  Root links: {root_links}")

def print_joint_info(urdf_data, joint_name):
    """
    Print detailed information about a specific joint.
    
    Args:
        urdf_data (dict): Parsed URDF data from parse_urdf()
        joint_name (str): Name of the joint
    """
    if urdf_data is None or joint_name not in urdf_data['joints']:
        print(f"Joint '{joint_name}' not found")
        return
    
    joint = urdf_data['joints'][joint_name]
    print(f"Joint: {joint_name}")
    print(f"  Type: {joint['type']}")
    print(f"  Parent: {joint['parent']}")
    print(f"  Child: {joint['child']}")
    print(f"  Origin XYZ: {joint['origin']['xyz']}")
    print(f"  Origin RPY: {joint['origin']['rpy']}")
    
    if joint['axis']:
        print(f"  Axis: {joint['axis']}")
    
    if joint['limits']:
        print(f"  Limits: {joint['limits']}")

def print_link_info(urdf_data, link_name):
    """
    Print detailed information about a specific link.
    
    Args:
        urdf_data (dict): Parsed URDF data from parse_urdf()
        link_name (str): Name of the link
    """
    if urdf_data is None or link_name not in urdf_data['links']:
        print(f"Link '{link_name}' not found")
        return
    
    link = urdf_data['links'][link_name]
    print(f"Link: {link_name}")
    print(f"  Mass: {link['mass']}")
    print(f"  Inertial Origin XYZ: {link['inertial_origin']['xyz']}")
    print(f"  Inertial Origin RPY: {link['inertial_origin']['rpy']}")
    print(f"  Inertia: {link['inertia']}")

# Example usage
if __name__ == "__main__":
    # Parse the URDF file
    urdf_file = "g1_29dof_rev_1_0.urdf"
    urdf_data = parse_urdf(urdf_file)
    
    if urdf_data:
        print("URDF parsed successfully!")
        print_urdf_summary(urdf_data)
        
        # Print information about specific joints/links
        print("\n" + "="*50)
        print_joint_info(urdf_data, "mid360_joint")
        
        print("\n" + "="*50)
        print_joint_info(urdf_data, "left_hip_pitch_joint")
        
        # Get specific offsets
        print("\n" + "="*50)
        mid360_offset = get_joint_offset(urdf_data, "mid360_joint")
        if mid360_offset is not None:
            print(f"Mid360 joint offset from parent: {mid360_offset}")
        
        # Compute chain offset (example: from pelvis to mid360)
        chain_offset = compute_chain_offset(urdf_data, "mid360_joint", "pelvis")
        if chain_offset is not None:
            print(f"Cumulative offset from pelvis to mid360: {chain_offset}")
        
        # List all joints
        print("\n" + "="*50)
        print("All joints:")
        for joint_name in urdf_data['joints'].keys():
            offset = get_joint_offset(urdf_data, joint_name)
            print(f"  {joint_name}: {offset}")
        
        # Find joints by type
        print("\n" + "="*50)
        revolute_joints = find_joints_by_type(urdf_data, 'revolute')
        fixed_joints = find_joints_by_type(urdf_data, 'fixed')
        print(f"Revolute joints ({len(revolute_joints)}): {revolute_joints[:5]}...")  # Show first 5
        print(f"Fixed joints ({len(fixed_joints)}): {fixed_joints[:5]}...")  # Show first 5
        
        # Get all offsets from pelvis
        print("\n" + "="*50)
        all_offsets = get_all_offsets_from_base(urdf_data, 'pelvis')
        print("Sample offsets from pelvis:")
        sample_joints = ['mid360_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint', 'head_joint']
        for joint in sample_joints:
            if joint in all_offsets:
                print(f"  {joint}: {all_offsets[joint]}")
        print(f"Relative offset from mid360 to head_joint: {compute_relative_offset('head_joint', 'mid360_joint')}")
    else:
        print("Failed to parse URDF file") 