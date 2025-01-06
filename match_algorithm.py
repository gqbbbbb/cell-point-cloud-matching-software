import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pycpd import RigidRegistration, DeformableRegistration


def dynamic_max_distance(points):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest_distances = distances[:, 1]
    return nearest_distances.mean()

def exact_match_hard(df_origin, df_target):
    """
    匹配两组三维点，输出符合最大距离限制的唯一匹配点对。

    参数:
        df_origin (pd.DataFrame): 第一组点集，包含列 ['Id', 'X', 'Y', 'Z']。
        df_target (pd.DataFrame): 第二组点集，包含列 ['Id', 'X', 'Y', 'Z']。
        max_distance (float): 最大匹配距离限制。

    返回:
        pd.DataFrame: 包含匹配结果的 DataFrame，每行包括:
            ['Id_origin', 'X_origin', 'Y_origin', 'Z_origin',
             'Id_target', 'X_target', 'Y_target', 'Z_target']
    """
    # 检查列是否正确
    required_columns = {'Id', 'X', 'Y', 'Z'}
    if not required_columns.issubset(df_origin.columns) or not required_columns.issubset(df_target.columns):
        raise ValueError("输入的 DataFrame 必须包含列 ['Id', 'X', 'Y', 'Z']")

    # 提取坐标数据
    points_origin = df_origin[['X', 'Y', 'Z']].values
    points_target = df_target[['X', 'Y', 'Z']].values

    # 构建目标点集的 KD 树
    tree_target = cKDTree(points_target)
    tree_origin = cKDTree(points_origin)

    # 对每个原始点找到最近的目标点
    distances_to_target, indices_to_target = tree_target.query(points_origin)

    # 对目标点找到最近的原始点
    distances_to_origin, indices_to_origin = tree_origin.query(points_target)

    # 构造匹配结果
    matched_data = []
    matched_targets = set()  # 记录已匹配的目标点的索引
    matched_origins = set()  # 记录已匹配的原始点的索引

    for i, (dist, idx) in enumerate(zip(distances_to_target, indices_to_target)):
        # 检查是否越界（query 返回的超出范围点索引会是 len(points_target)）
        if idx >= len(points_target) or i in matched_origins or idx in matched_targets:
            continue

        # 检查双向最近邻关系，确保唯一性
        if indices_to_origin[idx] == i:
            matched_data.append({
                'Id_origin': df_origin.iloc[i]['Id'],
                'X_origin': df_origin.iloc[i]['X'],
                'Y_origin': df_origin.iloc[i]['Y'],
                'Z_origin': df_origin.iloc[i]['Z'],
                'Id_target': df_target.iloc[idx]['Id'],
                'X_target': df_target.iloc[idx]['X'],
                'Y_target': df_target.iloc[idx]['Y'],
                'Z_target': df_target.iloc[idx]['Z'],
            })
            # 记录已匹配的点
            matched_origins.add(i)
            matched_targets.add(idx)

    # 转换为 DataFrame
    matched_df = pd.DataFrame(matched_data)
    return matched_df


def exact_match(df1, df2):
    """
    在两组三维点之间进行匹配。

    参数:
    - df1: 第一组点的 DataFrame，包含列 ['Id', 'X', 'Y', 'Z']
    - df2: 第二组点的 DataFrame，包含列 ['Id', 'X', 'Y', 'Z']
    - max_distance: 匹配的最大距离，超过该距离不匹配

    返回:
    DataFrame，包含匹配结果，每一行包括 [Id_origin, X_origin, Y_origin, Z_origin, Id_target, X_target, Y_target, Z_target]
    """
    MAX = 99999999
    # 验证输入
    for df in [df1, df2]:
        if not all(col in df.columns for col in ['Id', 'X', 'Y', 'Z']):
            raise ValueError("输入的 DataFrame 必须包含列 ['Id', 'X', 'Y', 'Z']")

    # 计算动态最大距离
    points = np.array(df2.loc[:,['X','Y','Z']])
    max_distance = dynamic_max_distance(points)

    # 初始化结果列表
    matches = {'Id_origin': [],
            'X_origin': [],
            'Y_origin': [],
            'Z_origin': [],
            'Id_target': [],
            'X_target': [],
            'Y_target': [],
            'Z_target': []}

    # 复制输入 DataFrame，避免修改原始数据
    length1 = len(df1)
    length2 = len(df2)
    remaining_df1 = list(range(length1))
    remaining_df2 = list(range(length2))

    # 计算两组点的距离矩阵
    distance_matrix = np.array(
        [[np.linalg.norm(np.array(df1.iloc[i, 1:3]) - np.array(df2.iloc[j, 1:3])) for j in range(length2)] for i in
         range(length1)])

    while remaining_df1 and remaining_df2:
        # 找到最近的一对点的索引
        min_distance = np.min(distance_matrix)
        if min_distance > max_distance:
            # 如果最近距离超过最大距离，停止匹配
            break

        # 获取最近点对的索引
        i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

        # 获取匹配点的信息
        point1 = df1.iloc[i]
        point2 = df2.iloc[j]

        # 将匹配点添加到结果中
        matches['Id_origin'].append(point1['Id'])
        matches['X_origin'].append(point1['X'])
        matches['Y_origin'].append(point1['Y'])
        matches['Z_origin'].append(point1['Z'])
        matches['Id_target'].append(point2['Id'])
        matches['X_target'].append(point2['X'])
        matches['Y_target'].append(point2['Y'])
        matches['Z_target'].append(point2['Z'])

        # 从两组点中移除已匹配的点
        remaining_df1.remove(i)
        remaining_df2.remove(j)
        for xx in range(length1):
            distance_matrix[xx][j] = MAX
        for yy in range(length2):
            distance_matrix[i][yy] = MAX

    for i in remaining_df1:
        point1 = df1.iloc[i]
        matches['Id_origin'].append(point1['Id'])
        matches['X_origin'].append(point1['X'])
        matches['Y_origin'].append(point1['Y'])
        matches['Z_origin'].append(point1['Z'])
        matches['Id_target'].append('')
        matches['X_target'].append('')
        matches['Y_target'].append('')
        matches['Z_target'].append('')

    for i in remaining_df2:
        point2 = df2.iloc[i]
        matches['Id_origin'].append('')
        matches['X_origin'].append('')
        matches['Y_origin'].append('')
        matches['Z_origin'].append('')
        matches['Id_target'].append(point2['Id'])
        matches['X_target'].append(point2['X'])
        matches['Y_target'].append(point2['Y'])
        matches['Z_target'].append(point2['Z'])

    # 将匹配结果转换为 DataFrame 并返回
    return pd.DataFrame(matches)

def cpd_registration(origin_somas,detected_somas,non_rigid=True, outlier_weight=None):
    if outlier_weight is None:
        if non_rigid:
            reg = DeformableRegistration(X=detected_somas, Y=origin_somas)
        else:
            reg = RigidRegistration(X=detected_somas, Y=origin_somas)
    else:
        if non_rigid:
            reg = DeformableRegistration(X=detected_somas, Y=origin_somas, w=outlier_weight)
        else:
            reg = RigidRegistration(X=detected_somas, Y=origin_somas, w=outlier_weight)
    origin_somas_t, _ = reg.register()
    return origin_somas_t

def cpd(data_origin,data_target,non_rigid=False):
    origin_somas = np.array(data_origin.loc[:,['X','Y']])
    detected_somas = np.array(data_target.loc[:,['X','Y']])
    origin_somas = cpd_registration(origin_somas,detected_somas,False)
    if non_rigid:
        origin_somas = cpd_registration(origin_somas,detected_somas,non_rigid)
    result = data_origin.copy()
    result.loc[:,['X','Y']] = origin_somas
    return result

def cpd_fully_auto(data_origin,data_target):
    origin_somas = np.array(data_origin.loc[:, ['X', 'Y']])
    detected_somas = np.array(data_target.loc[:, ['X', 'Y']])
    result_counts= -1
    result_points = None
    for flip in (True, False):
        for scale in (0.25, 0.5, 1, 2):
            for angle in range(0,360,30):
                transformed_somas = transform_point_cloud(origin_somas,angle,scale,flip)
                transformed_somas = cpd_registration(transformed_somas,detected_somas,False)
                counts = exact_match_hard_for_points(transformed_somas, detected_somas)
                if counts > result_counts:
                    result_counts = counts
                    result_points = transformed_somas
    result = data_origin.copy()
    result.loc[:, ['X', 'Y']] = result_points
    return result

def transform_point_cloud(points, rotation_angle, scale_factor, flip = False):
    """
    对二维点云进行绕中心旋转和放缩变换。

    参数：
        points (np.array): 一个形状为 (N, 2) 的二维点云数组，每行表示一个点 (x, y)。
        rotation_angle (float): 旋转角度，单位为弧度（radians）。
        scale_factor (float): 放缩因子，大于1表示放大，小于1表示缩小。

    返回：
        np.array: 变换后的二维点云数组，形状为 (N, 2)。
    """
    # 检查输入的点云是否为二维数组
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("输入的点云必须是形状为 (N, 2) 的二维 np.array 数组")

    # 计算点云的中心
    center = np.mean(points, axis=0)

    # 将点云平移到原点
    translated_points = points - center

    # 镜像
    if flip:
        translated_points = translated_points * np.array([[-1,1]])

    # 构造旋转矩阵
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                 [sin_theta, cos_theta]])

    # 对点云进行旋转
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    # 对点云进行放缩
    scaled_points = rotated_points * scale_factor

    # 平移回原中心
    transformed_points = scaled_points + center

    return transformed_points

def exact_match_hard_for_points(points_origin, points_target):
    # 构建目标点集的 KD 树
    tree_target = cKDTree(points_target)
    tree_origin = cKDTree(points_origin)

    # 对每个原始点找到最近的目标点
    distances_to_target, indices_to_target = tree_target.query(points_origin)

    # 对目标点找到最近的原始点
    distances_to_origin, indices_to_origin = tree_origin.query(points_target)

    # 构造匹配结果
    matched_data = []
    matched_targets = set()  # 记录已匹配的目标点的索引
    matched_origins = set()  # 记录已匹配的原始点的索引

    for i, (dist, idx) in enumerate(zip(distances_to_target, indices_to_target)):
        # 检查是否越界（query 返回的超出范围点索引会是 len(points_target)）
        if idx >= len(points_target) or i in matched_origins or idx in matched_targets:
            continue

        # 检查双向最近邻关系，确保唯一性
        if indices_to_origin[idx] == i:
            matched_data.append({
                'Id_origin': i,
                'Id_target': idx
            })
            # 记录已匹配的点
            matched_origins.add(i)
            matched_targets.add(idx)

    return len(matched_data)

def match_local_region(data_origin,data_target):
    origin_somas = np.array(data_origin.loc[:, ['X', 'Y']])
    detected_somas = np.array(data_target.loc[:, ['X', 'Y']])