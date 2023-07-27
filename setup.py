from setuptools import setup, find_packages

setup(
    name="QuantumAI",
    version="1.0",
    description="Des",
    author="FK",
    license="MIT",
    url="https://github.com/sobiodum/QuantumAI",
    packages=find_packages(),
    install_requires=[
        "alpaca-trade-api>=3",
        "ccxt>=3",
        "exchange-calendars>=4",
        "jqdatasdk>=1",
        "pyfolio>=0.9",
        "pyportfolioopt>=1",
        "ray[default,tune]>=2",
        "scikit-learn>=1",
        # "stable-baselines3>=2.0.0a5[extra]",
        "stable-baselines3[extra]"
        "stockstats>=0.5",
        "wrds>=3",
        "yfinance>=0.2",
        "optuna"
    ],
    extras_require={
        "dev": [
            "black>=23",
            "isort>=5",
            "jupyter>=1",
            "mypy>=1",
            "pandas-stubs>=2",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords=["", ""],
)
