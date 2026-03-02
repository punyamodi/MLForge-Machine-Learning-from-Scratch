from setuptools import setup, find_packages

setup(
    name="mlforge",
    version="1.0.0",
    author="MLForge Contributors",
    description="Machine learning algorithms implemented from scratch using NumPy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
