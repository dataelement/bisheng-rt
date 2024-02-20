import json
import os
import shutil
import signal
from logging import getLogger
from typing import Dict, List

import requests
import settings
from cmd_manage import CmdManage
from redis_manager import redis_client

logger = getLogger(__name__)


class SFTManage(object):
    # cmd执行的输出目录，绝对路径
    ClientCliOutput = settings.CLIENT_CLI_OUTPUT

    # 执行的训练指令
    ClientCli = 'bisheng_ft'

    # 已发布模型的跟目录
    ModelRootPath = settings.MODEL_ROOT_DIR

    # job 运行状态
    class JobStatus:
        Finished = 'FINISHED'
        Failed = 'FAILED'
        Running = 'RUNNING'

    def __init__(self, job_id: str):
        self.job_id = job_id

        self.exit_code_key = f'{job_id}:exitcode'
        self.stdout_key = f'{job_id}:stdout'
        self.stderr_key = f'{job_id}:stderr'
        self.model_name_key = f'{job_id}:model'
        self.exec_lock_key = f'{job_id}:execlock'

        # 若目录不存在则新建
        os.makedirs(self.ClientCliOutput, exist_ok=True)
        # 创建指令执行路径
        self.job_exec_dir = os.path.join(self.ClientCliOutput, job_id)
        os.makedirs(self.job_exec_dir, exist_ok=True)

        # 指令生成预训练模型的输出目录
        self.model_output_dir = os.path.join(self.job_exec_dir, 'model_output')
        os.makedirs(self.model_output_dir, exist_ok=True)

        # 指令本身执行结果目录
        self.result_dir = os.path.join(self.job_exec_dir, 'job_result')
        os.makedirs(self.result_dir, exist_ok=True)
        self.pid_path = os.path.join(self.result_dir, 'pid')
        self.stdout_path = os.path.join(self.result_dir, 'stdout')
        self.stderr_path = os.path.join(self.result_dir, 'stderr')

        # 训练集文件的本地目录
        self.train_dir = os.path.join(self.job_exec_dir, 'train_data')
        os.makedirs(self.train_dir, exist_ok=True)

    def run_job(self, options: List, commands: Dict):
        logger.info(f'start run finetune_job job_id: {self.job_id},'
                    f' options: {options}, commands: {commands}')
        try:
            # 尝试设置锁，判断是否任务被取消
            lock_ret = self.set_exec_lock_key()
            if not lock_ret:
                # 任务被取消
                logger.info(f'job is already canceled, job_id: {self.job_id}')
                return
            # 解析参数适配本地指令执行
            self.parse_commands(commands)

            # 执行指令
            stdout_file = open(self.stdout_path, 'wb')
            stderr_file = open(self.stderr_path, 'wb')
            pid_file = open(self.pid_path, 'wb')
            logger.info(f'start cmd job_id: {self.job_id}, params: {commands}')
            # 分布式部署时，此处指令的pid需要挂载同一个任务输出目录
            pid, code = CmdManage.execute_cmd(self.ClientCli,
                                              options,
                                              commands,
                                              stdout=stdout_file,
                                              stderr=stderr_file,
                                              pid_file=pid_file)
            logger.info(f'over cmd id:{self.job_id} pid:{pid} code:{code}')
            # 写入执行结果到redis内
            with open(self.stdout_path, 'r') as f:
                exec_stdout = f.read()
            with open(self.stderr_path, 'r') as f:
                exec_stderr = f.read()
            self.write_result(code, exec_stdout, exec_stderr)
            logger.info(f'finish run job, job_id: {self.job_id}')
        except Exception as e:
            logger.error(f'cmd failed id: {self.job_id}', exc_info=True)
            # 将执行结果写到本地文件内
            self.write_result(1, '', str(e))
            raise Exception(e)
        finally:
            self.release_exec_lock_key()

    def cancel_job(self):
        pid, exit_code, stdout, stderr = self.get_result()
        if exit_code is not None:
            # 任务已执行结束，无需cancel
            return
        # 没有exitcode 但是有pid说明还在进行中
        if pid is not None:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        # 说明任务还未执行，可能在队列中或解析参数的过程
        self.set_exec_lock_key()

    def set_exec_lock_key(self):
        return redis_client.setNx(self.exec_lock_key, 1)

    def release_exec_lock_key(self):
        redis_client.delete(self.exec_lock_key)

    def delete_job(self, model_name: str):
        # 先尝试取消任务
        self.cancel_job()

        # 获取已发布模型的目录，若存在则删除对应目录
        model_dir = os.path.join(self.ModelRootPath, model_name)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        self.delete_result()

    def publish_job(self, model_name: str):
        # 将预训练生产的模型目录拷贝到正式目录
        model_output_dir = os.path.join(self.model_output_dir)
        # 需要发布到的目录
        publish_model_dir = os.path.join(self.ModelRootPath, model_name)
        if not os.path.exists(model_output_dir):
            raise Exception(f'model not found, [{model_output_dir}] not exist')
        if os.path.exists(publish_model_dir):
            raise Exception(f'model already exists, [{publish_model_dir}]')
        shutil.copytree(model_output_dir, publish_model_dir)
        # 存储发布的模型名称
        redis_client.set_no_expire(self.model_name_key, model_name)

    def cancel_publish_job(self, model_name: str):
        pub_model_name = redis_client.get(self.model_name_key)
        if pub_model_name != model_name:
            raise Exception(f'[{model_name}] not pub model:[{pub_model_name}]')
        publish_model_dir = os.path.join(self.ModelRootPath, model_name)
        if os.path.exists(publish_model_dir):
            shutil.rmtree(publish_model_dir)

    def get_job_status(self) -> (str, str):
        """
        获取任务执行状态
        :return: 运行状态, 原因
        """
        pid, exit_code, stdout, stderr = self.get_result()
        if exit_code is not None:
            if int(exit_code) == 0:
                return self.JobStatus.Finished, stdout
            return self.JobStatus.Failed, stderr
        if pid is not None:
            return self.JobStatus.Running, ''
        # 还未开始执行，队列中或解析参数中
        return self.JobStatus.Running, ''

    def get_job_log(self) -> str:
        log_path = os.path.join(self.model_output_dir, 'trainer_log.jsonl')
        if not os.path.exists(log_path):
            logger.info(f'log not found, [{log_path}] is not exist')
            return ''
        with open(log_path, 'r') as f:
            return f.read()

    def get_job_metrics(self) -> Dict:
        metrics_path = os.path.join(self.model_output_dir, 'all_results.json')
        if not os.path.exists(metrics_path):
            logger.info(f'metrics not found, [{metrics_path}] is not exist')
            return {}
        with open(metrics_path, 'r') as f:
            return json.load(f)

    def change_model_name(self, old_model_name, model_name):
        pub_model_name = redis_client.get(self.model_name_key)
        if pub_model_name != old_model_name:
            raise Exception(f'[{old_model_name}] is not'
                            f' published model:[{pub_model_name}]')
        old_model_dir = os.path.join(self.ModelRootPath, old_model_name)
        if not os.path.exists(old_model_dir):
            raise Exception(f'old model dir [{model_name}] not exist')
        new_model_dir = os.path.join(self.ModelRootPath, model_name)
        if os.path.exists(new_model_dir):
            raise Exception(f'new model dir [{model_name}] already exist')
        os.rename(old_model_dir, new_model_dir)
        redis_client.set_no_expire(self.model_name_key, model_name)

    def get_result(self) -> (int, int, str, str):
        pid = None
        if os.path.exists(self.pid_path):
            pid = self.read_file(self.pid_path)
        exit_code = redis_client.get(self.exit_code_key)
        stdout = redis_client.get(self.stdout_key)
        stderr = redis_client.get(self.stderr_key)
        return pid, exit_code, stdout, stderr

    def write_result(self, exit_code: int, stdout: str, stderr: str):
        redis_client.set_no_expire(self.exit_code_key, str(exit_code))
        redis_client.set_no_expire(self.stdout_key, stdout)
        redis_client.set_no_expire(self.stderr_key, stderr)

    def delete_result(self):
        redis_client.delete(self.exit_code_key)
        redis_client.delete(self.stdout_key)
        redis_client.delete(self.stderr_key)
        shutil.rmtree(self.job_exec_dir)

    def read_file(self, file_path: str) -> str:
        with open(file_path, 'r') as f:
            return f.read().strip()

    def parse_commands(self, commands):
        # 将model_name 拼接为完整的绝对路径
        model_path = os.path.join(self.ModelRootPath,
                                  commands['model_name_or_path'])
        commands['model_name_or_path'] = model_path
        # 指定指令的输出路径
        commands['output_dir'] = self.model_output_dir

        # 将训练文件下载到本地 并重新拼接命令参数
        dataset_list = commands['dataset'].split(',')
        local_dataset = []
        for file_url in dataset_list:
            local_dataset.append(self.download_train_file(file_url))
        commands['dataset'] = ','.join(local_dataset)

    def download_train_file(self, file_url: str) -> str:
        """ 将远端的训练文件下载到本地 """
        local_file_path = os.path.join(self.train_dir,
                                       os.path.basename(
                                           file_url.split('?')[0]))
        res = requests.get(file_url)
        if res.status_code != 200:
            raise Exception(f'fail to download file from [{file_url}]')
        with open(local_file_path, 'wb') as f:
            f.write(res.content)
        return local_file_path
