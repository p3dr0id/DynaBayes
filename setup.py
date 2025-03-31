from setuptools import setup, find_packages

setup(
    name='dynabayes',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'pandas'],
    author='Pedro D. Pinto',
    description='dynamic bayesian inference for coupled phase oscillators',
    url='https://github.com/p3dr0id/DynaBayes',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
