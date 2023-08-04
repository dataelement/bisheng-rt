# flake8: noqa
import os


def _get_user_home():
    return os.path.expanduser('~')


def _get_ppnlp_home():
    if 'PPNLP_HOME' in os.environ:
        home_path = os.environ['PPNLP_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError(
                    'The environment variable PPNLP_HOME {} is not a directory.'
                    .format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddlenlp')


def _get_sub_home(directory, parent_home=_get_ppnlp_home()):
    home = os.path.join(parent_home, directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


USER_HOME = _get_user_home()
PPNLP_HOME = _get_ppnlp_home()
MODEL_HOME = _get_sub_home('models')
HF_CACHE_HOME = os.environ.get('HUGGINGFACE_HUB_CACHE', MODEL_HOME)
DATA_HOME = _get_sub_home('datasets')
PACKAGE_HOME = _get_sub_home('packages')
FAILED_STATUS = -1
SUCCESS_STATUS = 0

LEGACY_CONFIG_NAME = 'model_config.json'
CONFIG_NAME = 'config.json'
