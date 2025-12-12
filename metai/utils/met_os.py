import os
from typing import List, Union

def scan_directory(folder_path: str, level: int = 1, return_full_path: bool = False) -> Union[str, List[str]]:
    """
    递归扫描目录结构，返回子目录列表或单个目录名。

    注意：该函数为了兼容历史行为，返回类型并不总是列表：
    - 当 `level <= 0` 且 `return_full_path=False` 时，返回 **单个字符串**（当前目录名）。
    - 其他情况下通常返回 **字符串列表**。

    Args:
        folder_path: 要扫描的目录路径。
        level: 扫描深度。
            - `level <= 0`: 不展开子目录，仅返回当前目录（见返回值说明）。
            - `level == 1`: 返回当前目录下的直接子目录。
            - `level > 1`: 递归展开，返回“恰好第 `level` 层”的子目录列表（扁平列表）。
              例如：`level=2` 返回二级子目录（孙目录），不包含一级子目录本身。
        return_full_path: 返回值是否包含完整路径。
            - True: 返回完整路径（例如 `/a/b/c`）
            - False: 返回目录名（例如 `c`）

    Returns:
        - `level <= 0`:
            - `return_full_path=True` 时返回 `List[str]`（仅包含 `folder_path` 本身）
            - `return_full_path=False` 时返回 `str`（`os.path.basename(folder_path)`）
        - `level >= 1`: 返回 `List[str]`（目录列表；`level > 1` 时为扁平列表）
    """
    if level <= 0:
        return [folder_path] if return_full_path else os.path.basename(folder_path)
    
    # 读取当前目录下的直接子目录并排序（只保留目录，不包含文件）
    try:
        folders = sorted([
            os.path.join(folder_path, item) 
            for item in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, item))
        ])
    except (OSError, PermissionError):
        return []
    
    if level == 1:
        # level=1：只返回当前层级的子目录（非递归）
        if return_full_path:
            return folders
        else:
            return [os.path.basename(folder) for folder in folders]
    else:
        # level>1：递归展开，并合并成一个扁平列表
        result = []
        for folder in folders:
            sub_result = scan_directory(folder, level - 1, return_full_path)
            if isinstance(sub_result, list):
                result.extend(sub_result)
            else:
                result.append(sub_result)
        return result