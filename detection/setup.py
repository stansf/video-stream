"""Package for detection utils for video streaming service."""
from pathlib import Path
from setuptools import setup, find_namespace_packages

NAME = Path(__file__).absolute().parent.name

setup(
    name='vs-' + NAME,
    author='Stanislav Fateev',
    author_email='fateevsstanislav@gmail.com',
    description=__doc__,
    # packages=find_packages(include=('detection')),
    python_requires='>=3.8',
    zip_safe=False,
    include_package_data=True
)
