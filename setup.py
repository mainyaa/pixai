#  pixai

import codecs
import re
from os.path import (
    abspath,
    dirname,
    join,
)
try:
    from setuptools import setup
    from setuptools import find_packages
except ImportError:
    from distutils.core import setup


with codecs.open('pixai/task.py', 'r', 'utf-8') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)

if not version:
    raise RuntimeError('Cannot find version information')

def read_requirements(path):
    real_path = join(dirname(abspath(__file__)), path)
    with open(real_path) as f:
        reqs = f.readlines()
        return list(reqs)



setup(
    name='pixai',
    version=version,
    author='mainya',
    author_email='mainya@gmail.com',
    url='https://github.com/mainyaa/pixai',
    bugtrack_url='https://github.com/mainyaa/pixai/issues',
    license='MIT',
    packages=find_packages(include=['pixai', 'pixai.*']),
    entry_points={
        'console_scripts': [
            'pixai = pixai.evalute.__main__:main',
        ],
    },
    description='Video Super-Resolution Using a Generative Adversarial Network',
    long_description=codecs.open('README.md', encoding='utf-8').read(),
    #install_requires=['tensorflow', 'scikit-image'],
    platforms='any',
    #install_requires=read_requirements('requirements.txt'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
