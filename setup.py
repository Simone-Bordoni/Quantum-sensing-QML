"""Setup script for the Quantum Sensing Optimization Library."""

from setuptools import setup, find_packages

setup(
    name="qsopt",
    version="0.1.0",
    description="Quantum Sensing Optimization Library using JAX and QuTiP",
    author="Simone Bordoni",
    author_email="simone.bordoni@uniroma1.it",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.13",
    install_requires=[
        "numpy>=2.3.2",
        "jax>=0.7.1",
        "jaxlib>=0.7.1",
        "qutip>=5.2.1",
        "qutip-jax>=0.1.1",
        "optax>=0.2.5",
        "matplotlib>=3.10.6",
        "scipy>=1.16.1",
        "ml-dtypes>=0.5.3",
        "opt-einsum>=3.4.0",
        "chex>=0.1.91"
    ],
    extras_require={
        "dev": [
            "pytest>=8.4.0",
            "pytest-cov>=6.0.0",
            "coverage>=6.0.0",
            "pylint>=3.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "mypy>=1.11.0",
            "jupyter",
            "ipykernel"
        ],
        "test": [
            "pytest>=8.4.0",
            "pytest-cov>=6.0.0",
            "coverage>=6.0.0",
            "flake8>=7.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "mypy>=1.11.0"
        ]
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="quantum sensing, optimization, quantum optics, jax, qutip",
    long_description_content_type="text/markdown",
)
