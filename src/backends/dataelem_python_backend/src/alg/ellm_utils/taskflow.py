# flake8: noqa
import threading
import warnings

from .information_extraction import UIETask

warnings.simplefilter(action='ignore',
                      category=Warning,
                      lineno=0,
                      append=False)

TASKS = {
    'information_extraction': {
        'models': {
            'uie-x-base': {
                'task_class': UIETask,
                'hidden_size': 768,
                'task_flag': 'information_extraction-uie-x-base',
            },
        }
    },
}

support_schema_list = [
    'uie-x-base',
]

support_argument_list = [
    'uie-x-base',
]


class Taskflow(object):
    """
    The Taskflow is the end2end interface that could convert the raw text to model result, and decode the model result to task result. The main functions as follows:
        1) Convert the raw text to task result.
        2) Convert the model to the inference model.
        3) Offer the usage and help message.
    Args:
        task (str): The task name for the Taskflow, and get the task class from the name.
        model (str, optional): The model name in the task, if set None, will use the default model.
        mode (str, optional): Select the mode of the task, only used in the tasks of word_segmentation and ner.
            If set None, will use the default mode.
        device_id (int, optional): The device id for the gpu, xpu and other devices, the defalut value is 0.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.

    """
    def __init__(self,
                 task,
                 model=None,
                 mode=None,
                 device_id=0,
                 from_hf_hub=False,
                 **kwargs):
        assert task in TASKS, f'The task name:{task} is not in Taskflow list, please check your task name.'
        self.task = task

        tag = 'models'
        self.model = model

        if self.model is not None:
            assert self.model in set(TASKS[task][tag].keys(
            )), f'The {tag} name: {model} is not in task:[{task}]'

        if 'task_priority_path' in TASKS[self.task][tag][self.model]:
            self.priority_path = TASKS[self.task][tag][
                self.model]['task_priority_path']
        else:
            self.priority_path = None

        config_kwargs = TASKS[self.task][tag][self.model]
        kwargs['device_id'] = device_id
        kwargs.update(config_kwargs)
        self.kwargs = kwargs
        task_class = TASKS[self.task][tag][self.model]['task_class']
        self.task_instance = task_class(model=self.model,
                                        priority_path=self.priority_path,
                                        from_hf_hub=from_hf_hub,
                                        **self.kwargs)
        task_list = TASKS.keys()
        Taskflow.task_list = task_list

        self._lock = threading.Lock()

    def __call__(self, *inputs):
        """
        The main work function in the taskflow.
        """
        results = self.task_instance(inputs)
        return results

    def preprocess(self, *inputs):
        """
        The main work function in the taskflow.
        """
        results = self.task_instance._preprocess(inputs)
        return results

    def postprocess(self, inputs):
        """
        The main work function in the taskflow.
        """
        results = self.task_instance._postprocess(inputs)
        return results

    def run_model(self, inputs):
        """
        The main work function in the taskflow.
        """
        results = self.task_instance._run_model(inputs)
        return results

    def help(self):
        """
        Return the task usage message.
        """
        return self.task_instance.help()

    def task_path(self):
        """
        Return the path of current task
        """
        return self.task_instance._task_path

    @staticmethod
    def tasks():
        """
        Return the available task list.
        """
        task_list = list(TASKS.keys())
        return task_list

    def from_segments(self, *inputs):
        results = self.task_instance.from_segments(inputs)
        return results

    def interactive_mode(self, max_turn):
        with self.task_instance.interactive_mode(max_turn):
            while True:
                human = input('[Human]:').strip()
                if human.lower() == 'exit':
                    exit()
                robot = self.task_instance(human)[0]
                print('[Bot]:%s' % robot)

    def set_schema(self, schema):
        self.task_instance.set_schema(schema)

    def set_argument(self, argument):
        self.task_instance.set_argument(argument)
