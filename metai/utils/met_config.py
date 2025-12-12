"""
MetAI 配置管理模块。

提供统一的配置加载 / 保存与校验能力，支持：
- YAML 配置文件读取与写入
- 默认值与全局单例配置
- 配置项校验（路径 / 日期格式等）

"""

import yaml
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class MetConfig:
    """
    MetAI 配置对象。

    用于管理项目运行所需的基础配置（路径、文件命名格式等）。
    """
    
    # 数据集根目录路径
    root_path: str = "/home/dataset-local/SevereWeather_AI_2025"
    
    # GIS / DEM 地理数据路径
    gis_data_path: str = "/home/dataset-local/SevereWeather_AI_2025/dem"

    # 文件日期格式（`datetime.strftime` 格式）
    file_date_format: str = "%m%d-%H%M"

    # NWP 文件前缀
    nwp_prefix: str = "RRA"
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'MetConfig':
        """
        从 YAML 配置文件创建配置对象。

        Args:
            config_path: YAML 配置文件路径。

        Returns:
            MetConfig: 配置对象；当文件不存在或读取失败时返回默认配置。
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            print(f"[ERROR] 配置文件不存在: {config_path}")
            return cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 创建配置对象并更新值
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"[ERROR] 未知配置项: {key}")
            
            return config
            
        except Exception as e:
            print(f"[ERROR] 读取配置文件失败: {e}")
            return cls()
    
    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> 'MetConfig':
        """
        加载配置。
        
        Args:
            config_path: 配置文件路径；若为 None 则按默认路径顺序自动查找 `config.yaml`。
            
        Returns:
            MetConfig: 配置对象。
        """
        # 解析配置文件路径
        if config_path is None:
            config_file = "config.yaml"
            
            # 默认查找顺序（按优先级从高到低）
            default_paths = [
                Path.cwd() / config_file,
                Path.cwd() / "metai" / config_file,
                Path(__file__).parent.parent / config_file,
            ]
            
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break
        
        if config_path:
            config = cls.from_file(config_path)
            print(f"[INFO] 从配置文件加载配置: {config_path}")
        else:
            config = cls()
            print("[INFO] 使用默认配置")
        
        return config
    
    def save(self, config_path: Union[str, Path]) -> bool:
        """
        保存配置到 YAML 文件。

        Args:
            config_path: 配置文件路径。

        Returns:
            bool: 保存成功返回 True，否则返回 False。
        """
        config_path = Path(config_path)
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 序列化：导出非私有、非可调用属性
            config_data = {}
            for attr_name in dir(self):
                if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                    config_data[attr_name] = getattr(self, attr_name)
            
            # 写入 YAML 文件
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            print(f"[INFO] 配置已保存到: {config_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 保存配置文件失败: {e}")
            return False
    
    def get_root_path(self) -> str:
        """获取数据集根目录路径。"""
        return self.root_path
    
    def get_date_format(self) -> str:
        """获取日期格式。"""
        return self.file_date_format
    
    def get_nwp_prefix(self) -> str:
        """获取 NWP 文件前缀。"""
        return self.nwp_prefix
    
    def validate(self) -> bool:
        """
        验证配置的有效性。

        Returns:
            bool: 校验通过返回 True；否则返回 False，并输出错误信息。
        """
        errors = []
        
        # 必填项校验
        if not self.root_path:
            errors.append("root_path 不能为空")
        
        # 日期格式校验
        try:
            from datetime import datetime
            test_date = datetime(2024, 1, 1, 12, 0)
            test_date.strftime(self.file_date_format)
        except ValueError as e:
            errors.append(f"日期格式无效: {e}")
        
        if errors:
            for error in errors:
                print(f"[ERROR] 配置验证错误: {error}")
            return False
        
        return True


# 全局配置实例（懒加载）
_config: Optional[MetConfig] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> MetConfig:
    """
    获取全局配置实例（懒加载）。
    
    Args:
        config_path: 配置文件路径；若为 None 则按默认路径顺序自动查找 `config.yaml`。
        
    Returns:
        MetConfig: 配置对象。
    """
    global _config
    
    # 如果显式指定了 config_path，或尚未加载，则重新加载配置
    if config_path is not None or _config is None:
        _config = MetConfig.load(config_path)
        if not _config.validate():
            print("[ERROR] 配置验证失败，使用默认配置")
            _config = MetConfig()
    
    return _config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> MetConfig:
    """
    重新加载配置（强制刷新全局配置）。
    
    Args:
        config_path: 配置文件路径；若为 None 则按默认路径顺序自动查找 `config.yaml`。
        
    Returns:
        MetConfig: 新的配置对象。
    """
    global _config
    _config = MetConfig.load(config_path)
    return _config


def create_default_config(config_path: Union[str, Path]) -> bool:
    """
    创建默认配置文件。
    
    Args:
        config_path: 配置文件路径。
        
    Returns:
        bool: 是否创建成功。
    """
    config = MetConfig()
    return config.save(config_path)
