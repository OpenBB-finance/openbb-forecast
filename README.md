# OpenBB Forecast Extension

This extension provides Forecasting (FR) tools for the OpenBB SDK.

Features of the FR extension include various forecasting tools and models.

## Installation

Install this extension into your existing OpenBB Platform environment.

### On Windows, Linux and macOS with x86_64 CPUs

Activate your openbb environment and run the following command:

```bash
pip install git+https://github.com/OpenBB-finance/openbb-forecast.git
```

### On macOS with Apple Silicon chips

The installation is slightly different on modern macs.

Only [conda](https://docs.anaconda.com/free/miniconda/miniconda-install/) environment is supported because LightGBM does not provide arm64 binaries in the pypi package.
Apple Silicon users need to rely on the community build of LightGBM distributed via the [conda-forge](https://conda-forge.org/) channel.

You need to create a new conda environment if you are not already using conda for your openbb.
You can do this by running the following commands:

```bash
conda create -n obb python=3.10
conda activate obb
pip install openbb
```

Otherwise, you are good to continue.

First, install LightGBM via conda into your openbb conda environment:

```bash
conda install -c conda-forge lightgbm
```

Then, install the extension with the following command:

```bash
pip install git+https://github.com/OpenBB-finance/openbb-forecast.git
```

## Usage examples

You can find examples in command docstrings. Example `help(obb.forecast.statistical.mstl)`.

## References

For development please check [Contribution Guidelines](https://github.com/OpenBB-finance/OpenBBTerminal/blob/develop/openbb_platform/CONTRIBUTING.md).

OpenBB Platform Documentation available [here](https://docs.openbb.co/platform).
