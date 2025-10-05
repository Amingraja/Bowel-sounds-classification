from setuptools import setup, find_packages


setup(
    name="bowel-sound-classification",
    version="0.1.0",
    author="Amin Graja",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "datasets==4.1.1",
        "matplotlib==3.7.1",
        "numpy==1.25.1",
        "pandas==2.3.3",
        "pydub==0.25.1",
        "PyYAML==6.0.1",
        "scikit_learn==1.7.2",
        "seaborn==0.13.2",
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "transformers==4.27.3"
    ],
)

