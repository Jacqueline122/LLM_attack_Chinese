from setuptools import setup, find_packages

setup(
    name='LLM_attack_Chinese',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'accelerate',
        'transformers',
        'torch',
        'protobuf',
        'sentencepiece'
    ],
)
