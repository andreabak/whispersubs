# WhisperSubs

[![CI: pre-commit](https://github.com/andreabak/whispersubs/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/andreabak/whispersubs/actions)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Build: Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Mypy: checked](https://img.shields.io/badge/mypy-checked-2A6DB2.svg)](https://mypy-lang.org/)
[![License: LGPL-3.0](https://img.shields.io/github/license/andreabak/whispersubs)](https://github.com/andreabak/whispersubs/blob/main/LICENSE)

Generate subtitles for your video or audio files using the power of AI.

## Installation

Prerequisites:
- git for cloning this repo
- Python 3.8 or higher, with pip

For faster results, it is recommended to use a GPU with CUDA support.

Install the latest version from the repository:
```shell
pip install 'git+https://github.com/andreabak/whispersubs.git'
```
Or with CUDA libraries for GPU acceleration:
```shell
pip install 'whispersubs[cuda] @ git+https://github.com/andreabak/whispersubs.git'
```
Or using pipx to install globally in an isolated environment:
```shell
pipx install 'git+https://github.com/andreabak/whispersubs.git'
```
Or with pipx and CUDA libraries for GPU acceleration:
```shell
pipx install 'whispersubs[cuda] @ git+https://github.com/andreabak/whispersubs.git'
```

## Usage
WhisperSubs can be used from the command line. The basic usage requires an input file, which will be the source for the transcription. The output will be a subtitle file in the SRT format in the same directory as the input file. The language will be automatically detected.
```shell
whispersubs <input_file>
```

For a full list of options, run:
```shell
whispersubs --help
```
