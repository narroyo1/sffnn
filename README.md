# Stochastic Feedforward Neural Network

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [License](#license)

## Installation
### Clone
Start by making making a local clone of the repo.
```shell
git clone https://github.com/narroyo1/sffnn.git
```
### Setup
#### Using Anaconda
Create the environment.
```shell
conda env create -f environment.yml
```

Then activate it.
```shell
conda activate sffn
```

## Usage
### Using VS Code
Open `main.py` and run.

### Tensorboard visualization
On the same directory as `main.py` run:
```shell
tensorboard --logdir=runs
```

Then open `http://localhost:6006/` on a browser.

## Documentation

A paper throughly explaining the method is located at [docs/sffnn.md](docs/sffn.md).

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
