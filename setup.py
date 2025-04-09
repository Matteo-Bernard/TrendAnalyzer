from setuptools import setup, find_packages

setup(
    name="TrendAnalyzer",
    version="0.0.1",
    description="Outils de calcul de tendances boursières",
    author="Matteo Bernard",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
    ],
    include_package_data=True,
)
