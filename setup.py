#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup
import profK_libraries

# Taken from:
# https://packaging.python.org/en/latest/tutorials/packaging-projects/
#with open("README.md", "r") as fh:
    #long_description = fh.read()

setup(
    name='Modules',
    version=profK_libraries.__version__,
    description='Module for Statistical Methods',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=httpimport.__author__,
    author_email='fkauffeldt@web.de',
    url=httpimport.__github__,
    py_modules=['profK_libraries'],
    classifiers=[
        'Development Status :: 6 - Mature',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Software Development :: Testing',
    ],
    keywords=[
        'import',
        'loader',
        'memory',
        'http',
        'network'],
)

