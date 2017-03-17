import os
import sys
from distutils.core import setup

from setuptools import find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'malmopy'))
from version import VERSION

extras = {
    'chainer': ['chainer>=1.21.0'],
    'gym': ['gym[atari]>=0.7.0'],
    'tensorflow': ['tensorflow'],
}

# Meta dependency groups.
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

setup(
    name='malmopy',
    version=VERSION,

    packages=[package for package in find_packages()
              if package.startswith('malmopy')],

    url='https://github.com/Microsoft/malmo-challenge',
    license='MIT',
    author='Microsoft Research Cambridge',
    author_email='',
    description='Malmo Collaborative AI Challenge task and example code',
    install_requires=['future', 'numpy>=1.11.0', 'six>=0.10.0', 'pandas', 'Pillow'],
    extras_require=extras
)
