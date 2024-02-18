from celery import Celery
from settings import settings
from sft_manage import SFTManage

app = Celery('finetune_tasks', broker=settings.redis_url)


@app.task()
def create_job_task(job_id: str, options: list, commands: dict):
    sft_obj = SFTManage(job_id)
    sft_obj.run_job(options, commands)
