from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tfwn',
    version='1.0.1',
    description='Weight normalization layer for TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/shkarupa-alex/tfwn',
    author='Shkarupa Alex',
    author_email='shkarupa.alex@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        # https://github.com/tensorflow/tensorflow/issues/7166
        # 'tensorflow>=1.12.0',
    ],
    extras_require={
        'tf_cpu': ['tensorflow>=1.12.0'],
        'tf_gpu': ['tensorflow-gpu>=1.12.0'],
    },
    test_suite='nose.collector',
    tests_require=[
        'nose',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
