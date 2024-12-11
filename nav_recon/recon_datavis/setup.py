from distutils.core import setup
from setuptools import find_packages

setup(
    version='0.0.1',
    scripts=[],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
)