# GlassDynamics: Ridge Regression for Glass Viscosity Prediction

## Bachelor's thesis in physics
### Università degli Studi di Trieste

The main goal of this project is to develop a reliable and interpretable predictive model for glass viscosity, moving beyond "black-box" machine learning to provide physically meaningful insights for materials science applications.


This repository implements a workflow that bridges the gap between empirical models and modern data science techniques. Specifically, it utilizes a custom **Ridge Regression** approach to predict the parameters of the physical **MYEGA equation** (Glass transition temperature $T_g$, Fragility index $m$, and high-temperature viscosity limit $\log_{10} \eta_{\infty}$).


Run the following scripts before running the notebooks:
```sh
sudo apt install python3.11
sudo apt install python3.11-venv
python3.11 -m venv ".venv"
source .venv/bin/activate
pip install .
```