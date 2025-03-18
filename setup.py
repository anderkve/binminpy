from setuptools import setup, find_packages

setup(
    name="binminpy",
    version="0.1.0",
    description="A package for binned and parallelised function optimisation.",
    author="Anders Kvellestad",
    author_email="anders.kvellestad@fys.uio.no",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/anderkve/binminpy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy>=2.1.3",
        "scipy>=1.15.2",
        "matplotlib>=3.10.1",
    ],
    extras_require={
        'iminuit': ['iminuit>=2.30.1'],
        'MPI': ['mpi4py>=4.0.3'],
    },
    license="GPLv3+",
)
