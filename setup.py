from setuptools import setup, find_packages

setup(
    name="pioneer-nn",
    version="0.0.1",
    description="An in silico framework to simulate AI-experiment cycles for iterative improvement of genomic deep learning models.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(),
    python_requires=">=3.6, <4",
    # Do not install tensorflow here, because might want to use tensorflow or
    # tensorflow-cpu.
    install_requires=[
    ],
    extras_require={
        "dev": [
            "black",  # styler
            "flake8",  # linter
        ],
    },
)
