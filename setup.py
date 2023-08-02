import os
from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as version:
    VERSION = version.read()

requirements_path = os.path.join(
    os.path.dirname(__file__), 'requirements.txt'
)

if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(
    name="toolbox",
    version=VERSION,
    description="DoSelect's internal python tools",
    author="Doselect",
    author_email="Doselect@infoedge.com",
    url="https://github.com/doselect/yolov7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements_path,
    zip_safe=False,
    keywords="doselect object detection tool",
    classifiers=[
        'Environment :: Web Environment',
        'Intended Audience :: DoSelect Developers',
        'License :: Copyright 2018 Doselect',
        'Framework :: FastApi',
        'Programming Language :: Python :: 3.9',
    ]
)