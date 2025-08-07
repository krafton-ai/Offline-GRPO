from setuptools import find_packages, setup

setup(
    name="offline-grpo",
    version="0.0.0",
    description="offline-grpo",
    author="Krafton",
    packages=find_packages(
        include=[
            "deepscaler",
        ]
    ),
    install_requires=[
        "google-cloud-aiplatform",
        "pylatexenc",
        "sentence_transformers",
        "tabulate",
        "math-verify[antlr4_9_3]==0.6.0",
        "flash_attn==2.7.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
