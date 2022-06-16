from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]


setup(
    name="tagifai",
    version=0.1,
    description="Classify machine learning projects.",
    author="Goku Mohandas",
    author_email="goku@madewithml.com",
    url="https://madewithml.com/",
    python_requires=">=3.7",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
)
