import os
import signal
import subprocess
from typing import BinaryIO, Dict, List


class CmdManage:

    @classmethod
    def execute_cmd(cls, cmd: str, options: List[str], commands: Dict,
                    stdout: BinaryIO,
                    stderr: BinaryIO,
                    pid_file: BinaryIO,
                    timeout: int = None) -> (int, int):

        cmd = cls.parse_cmd(cmd, options, commands)
        try:
            p = subprocess.Popen(
                cmd,
                shell=True,
                preexec_fn=os.setsid,
                stderr=stdout,
                stdout=stderr,
            )
            # 写pid到文件内
            if pid_file:
                pid_file.write(str(p.pid).encode())
                pid_file.close()
            # 等待执行完成
            exit_code = p.wait(timeout=timeout)
            return p.pid, exit_code
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            raise Exception(f'timeout in execute cmd, [{cmd}]')
        except Exception as e:
            raise Exception(f'err in execute cmd: [{e}]')

    @classmethod
    def parse_cmd(cls, cmd: str, options: List[str], commands: Dict) -> str:
        cmd = f'{cmd} {" ".join(options)}'
        for key, value in commands.items():
            cmd += f' --{key} {value}'
        return cmd
