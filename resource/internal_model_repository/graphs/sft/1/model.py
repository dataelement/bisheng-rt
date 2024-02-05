import json
from typing import Dict

import numpy as np
import triton_python_backend_utils as pb_utils
from celery_tasks import create_job_task
from sft_manage import SFTManage


def _get_np_input(request, name, has_batch=True):
    return pb_utils.get_input_tensor_by_name(request, name).as_numpy()


def _get_optional_params(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return json.loads(tensor.as_numpy()[0]) if tensor else {}


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.args = args
        self.request_handler: Dict[str, callable] = {
            '/v2.1/sft/job': self.create_job,
            '/v2.1/sft/job/cancel': self.cancel_job,
            '/v2.1/sft/job/delete': self.delete_job,
            '/v2.1/sft/job/publish': self.publish_job,
            '/v2.1/sft/job/publish/cancel': self.cancel_publish_job,
            '/v2.1/sft/job/status': self.get_job_status,
            '/v2.1/sft/job/log': self.get_job_log,
            '/v2.1/sft/job/metrics': self.get_job_metrics,
            '/v2.1/sft/job/model_name': self.change_model_name,
        }

    def execute(self, requests):
        # TODO zgq: 处理不同的路由
        responses = []
        for request in requests:
            self.logger.log_info(f'---- request: {request}')
            response = self.dispatch_request(request)
            responses.append(response)
        return responses

    def dispatch_request(self, request):
        data = None
        try:
            inp_str = _get_np_input(request, 'INPUT')[0]
            # 这部分就是输入的json数据，uri字段表示用户发起的uri信息
            payload = json.loads(inp_str)
            self.logger.log_info(f'==== request payload: {request}')
            uri = payload['uri']
            handler = self.request_handler.get(uri)
            if not handler:
                self.logger.error(f'invalid uri not find handler: {uri}')
                status_code = 400
                status_message = f'invalid uri not find handler: {uri}'
            else:
                status_code, status_message, data = handler(payload)
        except Exception as e:
            status_code = 400
            status_message = f'error in handle request: {str(e)}'
        result = {
            'status_code': status_code,
            'status_message': status_message,
            'data': data
        }
        result_arr = np.array([json.dumps(result)], dtype=np.object_)

        out_tensor_0 = pb_utils.Tensor('OUTPUT', result_arr)
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[out_tensor_0])
        return inference_response

    def create_job(self, payload):
        try:
            job_id = payload['job_id']
            options = payload['options']
            params = payload['params']
            # 发起异步任务
            create_job_task.delay(job_id, options, params)
        except Exception as e:
            return 400, f'fail to create job: {str(e)}', None
        return 200, 'success', None

    def cancel_job(self, payload):
        try:
            job_id = payload['job_id']
            sft_obj = SFTManage(job_id)
            sft_obj.cancel_job()
        except Exception as e:
            return 400, f'fail to cancel job: {str(e)}', None
        return 200, 'success', None

    def delete_job(self, payload):
        try:
            job_id = payload['job_id']
            model_name = payload['model_name']
            sft_obj = SFTManage(job_id)
            sft_obj.delete_job(model_name)
        except Exception as e:
            return 400, f'fail to delete job: {str(e)}', None
        return 200, 'success', None

    def publish_job(self, payload):
        try:
            job_id = payload['job_id']
            model_name = payload['model_name']
            sft_obj = SFTManage(job_id)
            sft_obj.publish_job(model_name)
        except Exception as e:
            return 400, f'fail to publish job: {str(e)}', None
        return 200, 'success', None

    def cancel_publish_job(self, payload):
        try:
            job_id = payload['job_id']
            model_name = payload['model_name']
            sft_obj = SFTManage(job_id)
            sft_obj.cancel_publish_job(model_name)
        except Exception as e:
            return 400, f'fail to cancel publish job: {str(e)}', None
        return 200, 'success', None

    def get_job_status(self, payload):
        try:
            job_id = payload['job_id']
            sft_obj = SFTManage(job_id)
            status, reason = sft_obj.get_job_status()
            data = {
                'status': status,
                'reason': reason
            }
        except Exception as e:
            return 400, f'fail to get job status: {str(e)}', None
        return 200, 'success', data

    def get_job_log(self, payload):
        try:
            job_id = payload['job_id']
            sft_obj = SFTManage(job_id)
            log = sft_obj.get_job_log()
            data = {
                'log_data': log
            }
        except Exception as e:
            return 400, f'fail to get job log: {str(e)}', None
        return 200, 'success', data

    def get_job_metrics(self, payload):
        try:
            job_id = payload['job_id']
            sft_obj = SFTManage(job_id)
            metrics = sft_obj.get_job_metrics()
            data = {
                'report': metrics
            }
        except Exception as e:
            return 400, f'fail to get job metrics: {str(e)}', None
        return 200, 'success', data

    def change_model_name(self, payload):
        try:
            job_id = payload['job_id']
            old_model_name = payload['old_model_name']
            new_model_name = payload['model_name']
            sft_obj = SFTManage(job_id)
            sft_obj.change_model_name(old_model_name, new_model_name)
        except Exception as e:
            return 400, f'fail to change model name: {str(e)}', None
        return 200, 'success', None

    def finalize(self):
        self.logger.log_info('finalize model')
