from setuptools import setup, find_packages

setup(
    name='bayesinf',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'pandas'],
    author='Pedro D. Pinto',
    description='Bayesian inference for coupled phase oscillators',
    url='https://github.com/p3dr0id/bayesinf',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
