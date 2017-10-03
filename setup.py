# setup.py
from setuptools import setup, find_packages
setup(name='yata',
      version='0.1',
      description='Data loader for deep learning',
      author='Yu Yin',
      author_email='yxonic@gmail.com',
      url='https://github.com/yxonic/ydata',
      install_requires=['pandas', 'six'],
      extras_require={
          'image': ['pillow'],
          'numpy': ['numpy'],
          'torch': ['torch']
      },
      packages=find_packages())
