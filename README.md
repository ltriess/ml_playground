# ML Playground

A repo to try out some ML concepts

## Installation

To recreate the conda environment from the `environment.yml` file, use the following command:
```bash
conda env create -f environment.yml
```

This will create a new conda environment with the same name and packages as specified in the `environment.yml` file.

Activate the environment with
```bash
conda activate ml_playground
```

## Setting Up Pre-commit Hooks

The package `pre-commit` is already contained in the conda environment. Otherwise run `pip install pre-commit` first.

Install `pre-commit` with
```bash
pre-commit install
```

To run `pre-commit` on all files, run
```bash
pre-commit run --all-files
```
