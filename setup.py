# setup.py
from setuptools import setup, find_packages

setup(
    name='alba_funcs',
    version='0.1',
    author = 'Yonatan Broder',
    description = 'analysis of XANES data of guanine acquired in ALBA 2022-2023',
    lab = 'dvir gur',
    install_requires = [
        'h5py==3.11.0',
        'matplotlib==3.9.2',
        'more_itertools==10.5.0',
        'numpy==2.1.1',
        'pandas==2.2.2',
        'Pillow==10.4.0',
        'plotly==5.22.0',
        'pyperclip==1.9.0',
        'scikit_learn==1.5.1',
        'scipy==1.14.1',
        'seaborn==0.13.2',
        'tqdm==4.66.5',
    ],
    python_requires = '>=3.9',
    packages=find_packages(),
)
