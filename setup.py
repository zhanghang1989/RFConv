##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import io
import os
import subprocess

from setuptools import setup, find_packages
import setuptools.command.develop 
import setuptools.command.install 

cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.0.0'
try:
    from datetime import date
    today = date.today()
    day = today.strftime("b%d%m%Y")
    version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'rfconv', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is rfconv version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

class install(setuptools.command.install.install):
    def run(self):
        create_version_file()
        setuptools.command.install.install.run(self)

class develop(setuptools.command.develop.develop):
    def run(self):
        create_version_file()
        setuptools.command.develop.develop.run(self)

readme = open('README.md').read()

requirements = [
    'Pillow',
    'scipy',
    'requests',
    'numpy',
    'tqdm',
    'nose',
    'torch>=1.4.0',
]

setup(
    name="rfconv",
    version=version,
    author="Hang Zhang",
    author_email="zhanghang0704@gmail.com",
    url="https://github.com/zhanghang1989/RFConv",
    description="Rectified Convolution",
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    install_requires=requirements,
    packages=find_packages(exclude=["scripts", "examples", "tests"]),
    package_data={'rfconv': [
        'LICENSE',
        'lib/*.h',
        'lib/*.cpp',
        'lib/*.cu',
    ]},
    cmdclass={
        'install': install,
        'develop': develop,
    },
)


