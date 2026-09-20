from setuptools import setup, find_packages

# Install the CUDA-specific PyTorch 2.x stack separately; see README.md.
requirements = [
    "tqdm",
    "kaldiio",
    "thop>=0.1.1",
]

setup(
    name="wesep",
    license="Apache-2.0",
    install_requires=requirements,
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
    entry_points={
        "console_scripts": [
            "wesep = wesep.cli.extractor:main",
        ],
    },
)
