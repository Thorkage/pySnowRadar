import re
from setuptools import find_packages, setup

with open('./pySnowRadar/__init__.py') as f:
    version = re.search(r"__version__ = \s*'([\d.*]+)'", f.read()).group(1)

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='pySnowRadar',
    version=version,
    author='Climate Research Division',
    author_email='',
    description='A Python3 package to process data from CRESIS SnowRadar systems',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kingjml/pySnowRadar',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires='>=3,<3.8',
    install_requires=[
        'gdal>=2.3.3',
        'h5py>=2.9',
        'matplotlib>=3.1',
        'numpy>=1.16',
        'pywavelets>=1',
        'shapely>=1.6.4',
        'scipy>=1.2.1',
        'pandas>=0.24.2',
        'geopandas>=0.4.1',
        'pyproj>=1.9.6',
    ],
    extras_require={
        'test': ['pytest', 'coverage'],
        'notebooks': ['jupyter']
    },
    keywords='snowradar',
    project_urls={
        'Source': 'https://github.com/kingjml/pySnowRadar',
        'Tracker': 'https://github.com/kingjml/pySnowRadar/issues',
    },
)
