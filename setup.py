from setuptools import setup, find_packages

setup(
    name="binminpy",
    version="0.1.0",
    description="A package for binned and parallelised sampling and optimisation.",
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
        "numpy>=1.25.2",
        "scipy>=1.9.3",
    ],
    extras_require={
        'iminuit': ['iminuit>=2.23.0'],
        'MPI': ['mpi4py>=3.1.1'],
    },
    license="GPLv3+",
)
