# Trajax Examples

This directory contains examples, benchmarks, and notebooks demonstrating the use of trajax.

## Structure

- **notebooks/**: Jupyter notebooks with tutorials and case studies
- **benchmarks/**: Performance benchmarks for different solvers

## Notebooks

The notebooks demonstrate various use cases:
- iLQR for trajectory optimization
- Constrained optimization with augmented Lagrangian
- Model Predictive Control
- Different solver comparisons

## Benchmarks

Performance benchmarks comparing:
- iLQR vs other solvers
- Different QP backends
- Linearization methods (finite differences vs autodiff)

## Running Examples

Install trajax with optional dependencies:
```bash
pip install -e ".[all]"
```

For notebooks, also install:
```bash
pip install jupyter matplotlib
```

Then run:
```bash
jupyter notebook examples/notebooks/
```
