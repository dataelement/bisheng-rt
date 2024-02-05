from celery import Celery
from sft_manage import SFTManage

import config

app = Celery('finetune_tasks', broker=config.CELERY_REDIS_URL)


@app.task()
def create_job_task(job_id: str, options: list, commands: dict):
    sft_obj = SFTManage(job_id)
    sft_obj.run_job(options, commands)
