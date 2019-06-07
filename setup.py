from setuptools import setup

setup(
   name='pal',
   version='0.3',
   description='Two adaptive reinforcement learning controllers that are controlling a pendulum together',
   author='Anonymous Author',
   author_email='anonymous@author.com',
   packages=['pal'],  # same as name
   install_requires=['numpy', 'matplotlib', 'scipy', 'keras', 'tensorflow'],
)
