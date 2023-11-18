# Copyright (c) 2023 pybackend libs Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import os

import setuptools


def _load_version() -> str:
    ROOT_PATH = os.path.dirname(__file__)
    version_path = os.path.join(ROOT_PATH, 'version.txt')
    version = open(version_path).readlines()[0].strip()
    return version


def read_requirements_file(filepath):
    with open(filepath) as fin:
        contents = fin.readlines()
    pkgs = [v for v in contents if not v.startswith('-') and v.strip()]
    return pkgs


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get('encoding', 'utf8')) as fp:
        return fp.read()


def get_package_data_files(package, data, package_dir=None):
    """
    Helps to list all specified files in package including files in directories
    since `package_data` ignores directories.
    """
    if package_dir is None:
        package_dir = os.path.join(*package.split('.'))
    all_files = []
    for f in data:
        path = os.path.join(package_dir, f)
        if os.path.isfile(path):
            all_files.append(f)
            continue
        for root, _dirs, files in os.walk(path, followlinks=True):
            root = os.path.relpath(root, package_dir)
            for file in files:
                file = os.path.join(root, file)
                if file not in all_files:
                    all_files.append(file)
    return all_files


extras = {}
REQUIRED_PACKAGES = read_requirements_file('requirements.txt')
PACKAGES = setuptools.find_packages(
    where='src',
    exclude=('examples*', 'tests*', 'applications*', 'model_zoo*'))

setuptools.setup(
    name='bisheng-pybackend-libs',
    version=_load_version(),
    author='DataElem Inc.',
    author_email='contact@dataelem.com',
    description='libraries for bisheng rt pybackend',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/dataelement/bisheng-rt/python/pybackend_libs',
    packages=PACKAGES,
    package_dir={'': 'src'},
    package_data={'': ['*.ttc']},
    setup_requires=[],
    install_requires=REQUIRED_PACKAGES,
    entry_points={},
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    license='Apache 2.0',
)
