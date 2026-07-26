from setuptools import setup, find_packages

setup(
    name="vagen",
    version="26.2.5",
    packages=find_packages(),
    install_requires=[
        "gym-sokoban",
        "gymnasium",
        "gymnasium[toy-text]",
        "fire>=0.7,<0.8",
        "fastapi>=0.116,<0.117",
        "httpx>=0.28,<0.29",
        "openai>=1.99,<2",
        "uvicorn<0.41",
    ],
    extras_require={
        "navigation": [
            "ai2thor==5.0.0",
        ],
    },
    python_requires=">=3.10",
)
