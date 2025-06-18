import yaml
import logging
import os

def load_config(config_path='config/config.yaml'):
    """Loads a YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary, or None if an error occurs.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"配置文件 {config_path} 加载成功。")
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件 {config_path} 未找到。")
        # Try to load from parent directory if in src for testing
        alt_config_path = os.path.join("..", config_path)
        try:
            with open(alt_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"备用配置文件 {alt_config_path} 加载成功。")
            return config
        except FileNotFoundError:
            print(f"错误: 备用配置文件 {alt_config_path} 也未找到。")
            return None
        except Exception as e:
            print(f"加载备用配置文件 {alt_config_path} 时出错: {e}")
            return None
    except yaml.YAMLError as e:
        print(f"解析配置文件 {config_path} 时出错: {e}")
        return None
    except Exception as e:
        print(f"加载配置文件 {config_path} 时发生未知错误: {e}")
        return None

def setup_logging(log_dir='logs', log_file_name='project.log', level=logging.INFO):
    """Sets up basic logging for the project.

    Args:
        log_file (str): Name of the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            # If running from main.py, logger might not be set up yet, so print
            print(f"Error creating log directory {log_dir}: {e}")
            # Optionally, re-raise or handle if critical
            return # Exit if log directory cannot be created

    full_log_path = os.path.join(log_dir, log_file_name)
    
    # Configure logging
    # Remove existing handlers if any to avoid duplicate logs on re-setup (e.g. in tests)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                        handlers=[
                            logging.FileHandler(full_log_path, mode='a'), # Append mode
                            logging.StreamHandler() # Also print to console
                        ])
    logging.info("日志记录已设置。")

# Example of another utility function (not strictly required by the prompt but good practice)
def ensure_dir(directory_path):
    """Ensures that a directory exists, creating it if necessary.

    Args:
        directory_path (str): The path to the directory.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logging.info(f"目录 {directory_path} 已创建。")
        except Exception as e:
            logging.error(f"创建目录 {directory_path} 时出错: {e}")
            # Depending on severity, might raise the exception or handle it

if __name__ == '__main__':
    # Test load_config
    print("--- 测试 load_config ---")
    # Create a dummy config for testing if it doesn't exist in expected paths
    dummy_config_content = """
api_key: 'TEST_KEY_FROM_UTILS_EXAMPLE'
tickers:
  - TST1
  - TST2
start_date: '2022-01-01'
train_end_date: '2022-12-31'
max_lag: 5
"""
    test_config_dir = 'config'
    test_config_file = os.path.join(test_config_dir, 'config.yaml')
    
    if not os.path.exists(test_config_dir):
        os.makedirs(test_config_dir)
        print(f"创建了测试目录 {test_config_dir}")

    if not os.path.exists(test_config_file):
        try:
            with open(test_config_file, 'w') as f_cfg:
                f_cfg.write(dummy_config_content)
            print(f"创建了用于测试的虚拟 {test_config_file} 文件。")
        except Exception as e:
            print(f"创建虚拟配置文件时出错: {e}")

    config_data = load_config(test_config_file) # Test with direct path
    if config_data:
        print("加载的配置数据:", config_data)
    else:
        print("未能加载配置数据。")
    
    # Test ensure_dir
    print("\n--- 测试 ensure_dir ---")
    test_data_dir = 'data_utils_test'
    ensure_dir(test_data_dir)
    if os.path.exists(test_data_dir):
        print(f"目录 {test_data_dir} 存在或已创建。")
        try:
            os.rmdir(test_data_dir) # Clean up
            print(f"测试目录 {test_data_dir} 已删除。")
        except OSError as e:
            print(f"删除测试目录 {test_data_dir} 时出错: {e} (可能是因为目录不为空)")

    # Test logging (output will go to console and project.log)
    print("\n--- 测试 setup_logging ---")
    setup_logging(log_dir='logs', log_file_name='utils_test.log') # Test with specific dir and file
    logging.info("这是一条来自 utils.py 测试的信息日志。")
    logging.warning("这是一条来自 utils.py 测试的警告日志。")