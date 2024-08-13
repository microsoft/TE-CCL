from setuptools import setup, find_packages

setup(
    name='teccl',
    version='1.0.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'teccl = teccl.__main__:main',
        ],
    },
    install_requires=[
        'dataclasses',
        'argcomplete',
        'gurobipy',
        'numpy',
        'seaborn'
    ],
    python_requires='>=3.6',
)