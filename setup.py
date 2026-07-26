from setuptools import setup, find_packages

setup(
    name="vagen",
    version="26.2.5",
    packages=find_packages(),
    install_requires=[
        "gym-sokoban",
        "gymnasium",
        "gymnasium[toy-text]",
        "fastapi>=0.116,<0.117",
        "httpx>=0.28,<0.29",
        "uvicorn<0.41",
    ],
    python_requires=">=3.10",
)
