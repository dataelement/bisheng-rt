import os

import yaml
from cryptography.fernet import Fernet
from pydantic import BaseSettings, validator

# 以下配置和镜像目录相关，不可改动！！！
# ָbisheng-ft命令的输出根目录
CLIENT_CLI_OUTPUT = '/opt/bisheng-rt/finetune_output'
# RT上已发布模型的根目录
MODEL_ROOT_DIR = '/opt/bisheng-rt/models/model_repository'

secret_key = 'TI31VYJ-ldAq-FXo5QNPKV_lqGTFfp-MIdbK2Hm5F1E='


def encrypt_token(token: str):
    return Fernet(secret_key).encrypt(token.encode())


def decrypt_token(token: str):
    return Fernet(secret_key).decrypt(token).decode()


class Settings(BaseSettings):
    # celery的broker；存储训练指令的执行结果
    redis_url: str = None

    @validator('redis_url')
    @classmethod
    def set_redis_url(cls, v: str):
        import re
        pattern = r'(?<=:)[^:]+(?=@)'  # 匹配冒号后面到@符号前面的任意字符
        match = re.search(pattern, v)
        if match:
            password = match.group(0)
            new_password = decrypt_token(password)
            new_redis_url = re.sub(pattern, f'{new_password}', v)
            return new_redis_url
        # 无账号密码
        return v


def load_settings_from_yaml(file_path: str) -> Settings:
    # Get current path
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Check if a string is a valid path or a file name
    if '/' not in file_path:
        file_path = os.path.join(current_path, file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        settings_dict = yaml.safe_load(f)

    for key in settings_dict:
        if key not in Settings.__fields__.keys():
            raise KeyError(f'Key {key} not found in settings')

    return Settings(**settings_dict)


# 在同级目录寻找配置文件。目前实现方案，sft相关代码配置都在此目录内
settings = load_settings_from_yaml('config.yaml')
