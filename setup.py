# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:45:35 2018

@author: Tom
"""

from setuptools import setup, find_packages

#import os
#
#packageList=['dataAnalysis']
#
#for package in os.listdir('src/dataAnalysis'):
#    packageList.append('dataAnalysis.'+package)

setup(name='dataAnalysis',
      version='1.0.dev1',
      description='Data Analysis Libraries',
      author='Tom Bennett-Lloyd',
      packages=find_packages("src"),
      package_dir={'': 'src'}
     )