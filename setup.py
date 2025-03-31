from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dynabayes',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas"
    ],
    author='Pedro D. Pinto',
    author_email='pedrodp42@gmail.com',
    description='Dynamic Bayesian Inference Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/p3dr0id/DynaBayes',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

