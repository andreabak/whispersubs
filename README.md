# WhisperSubs

[![PyPI: latest release](https://img.shields.io/pypi/v/whispersubs.svg?logo=pypi&label=PyPI)](https://pypi.org/project/whispersubs/)
[![PyPI: Python Version](https://img.shields.io/pypi/pyversions/whispersubs.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/whispersubs/)
[![CI: pre-commit](https://github.com/andreabak/whispersubs/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/andreabak/whispersubs/actions)  
[![Build: Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Mypy: checked](https://img.shields.io/badge/mypy-checked-2A6DB2.svg)](https://mypy-lang.org/)
[![License: LGPL-3.0](https://img.shields.io/github/license/andreabak/whispersubs)](https://github.com/andreabak/whispersubs/blob/main/LICENSE)

*Generate subtitles for your video or audio files using the power of AI.*

## Installation

Prerequisites:
- git for cloning this repo
- Python 3.8 or higher, with pip

For faster results, it is recommended to use a GPU with CUDA support. Running transcription on the CPU is significantly slower (up to 10~20x slower, depending on the hardware).  
The optional dependencies required for GPU acceleration can be installed with the `[cuda]` extra when installing the package.

### Install latest release
Use the following to install the latest release in an isolated environment using [`pipx`](https://pipx.pypa.io), with CUDA libraries for GPU acceleration:
```shell
pipx install whispersubs[cuda]
```

N.B. omit the `[cuda]` extra if you don't have a GPU or don't want to use it.

Replace `pipx` with `pip` from the command above if you want to manually manage your own environment (or install user- or system-wide).

### Install repository version
You can install the latest version from this repository with the following command:
```shell
pipx install 'whispersubs[cuda] @ git+https://github.com/andreabak/whispersubs.git'
```
Usually this is going to be the same as the latest release, but it might contain some additional features or bugfixes that are not yet released.

## Usage
WhisperSubs can be used from the command line. The basic usage requires an input file, which will be the source for the transcription. The output will be a subtitle file in the SRT format in the same directory as the input file. The language will be automatically detected.
```shell
whispersubs <input_file>
```

For a full list of options, run:
```shell
whispersubs --help
```
