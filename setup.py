from setuptools import setup, find_packages

setup(
    name="TrendDistrib",
    version="0.0.1",
    description="Calcul de distribution des tendances boursières",
    author="Matteo Bernard",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "tempfile"
    ],
    include_package_data=True,
)
