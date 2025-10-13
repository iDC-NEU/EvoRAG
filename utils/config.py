import yaml
import os

# 你也可以写成单例，避免重复加载
class Config:
    def __init__(self):
        # 主配置文件路径（也可以改成传参）
        main_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'main_config.yaml')
        with open(main_config_path, 'r') as f:
            main_cfg = yaml.safe_load(f)

        config_file = main_cfg.get('config_file')
        if not config_file:
            raise ValueError("main_config.yaml中必须包含'config_file'字段")

        target_path = os.path.join(os.path.dirname(main_config_path), config_file)
        with open(target_path, 'r') as f:
            self.algorithm_config = yaml.safe_load(f)

        target_path = os.path.join(os.path.dirname(main_config_path), "path-local.yaml")
        with open(target_path, 'r') as f:
            self.path_config = yaml.safe_load(f)

    def get_algorithm(self):
        return self.algorithm_config
    
    def get_path(self):
        return self.path_config

# 全局单例对象
algorithm_config = Config().get_algorithm()
path_config = Config().get_path() 
