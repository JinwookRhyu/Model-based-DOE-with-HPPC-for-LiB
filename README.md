This repository contains the software for [Optimum Model-Based Design of Diagnostics Experiments (DOE) with Hybrid Pulse Power Characterization (HPPC) for Lithium-Ion Batteries] which can be used for performing model-based DOE for optimizing the HPPC protocol. This software is associated with the paper 'Optimum Model-Based Design of Diagnostics Experiments (DOE) with Hybrid Pulse Power Characterization (HPPC) for Lithium-Ion Batteries' by Jinwook Rhyu et al.

![alt text](https://github.com/JinwookRhyu/Model-based-DOE-with-HPPC-for-LiB/blob/main/graphical_abstract_v1.jpg?raw=true)

# Code

The software is performed in Python where design_generalized_optimal_HPPC.py and perform_MCMC.py are the main functions for performing model-based DoE and MCMC simulations, respectively.
MCMC_autocorrelation_help.py is a helper function for performing MCMC simulations.
extract_J1_J2_from_protocol.py is a function for calculating the two objective functions (f_uncertainty and f_time) when the protocol is given.
Figures.py is a function used to generate a figure for MCMC results.
plot_pareto.m is a code used to generate a figure for pareto plots.

design_generalized_optimal_HPPC.py requires pygmo2 installation (https://esa.github.io/pygmo2/install.html)
perform_MCMC.py requires emcee installation (https://emcee.readthedocs.io/en/stable/user/install/)

# Main updates for ver. 2

![alt text](https://github.com/JinwookRhyu/Model-based-DOE-with-HPPC-for-LiB/blob/main/graphical_abstract_v2.jpg?raw=true)

1. Added option to solve the optimization problem for f_uncertainty and f_time simultaneously using non-dominated sorting genetic algorithm (nsga2).
2. Added A-optimality for formulating f_uncertainty.
3. Accounted for multiscale structure of electrode particles (e.g., primary particle size for calculating A_electrode / A_particle. secondary particle size for diffusion terms).
4. Updated parameters for electron-coupled ion transfer model (e.g., pre-factor for exchange current density (k_0*), reorganization energy (\lambda)).
5. Reduced search space from [0, 10 Ohm.m^2] to [0, 0.2 Ohm.m^2] for R_f_c and R_f_a.
6. Corrected the missing (1/\sqrt{4 pi lambda}) term for ECIT current.
7. Changed the measurement error in the current from absolute to relative.
8. Increased the number of samples from 1,000 to 10,000 for calculating f_uncertainty.
9. Added option to construct a surrogate model with Gaussian process for calculating f_uncertainty.

# Folders

MCMC_results folder contains the MCMC results that were used to generate Figures 4, C1, and C2.
Pareto_results folder contains the model-based DoE results that were used to generate Figures 2 and 5.
amin_diffusion and diffusion_carelli_et_al folders contain the diffusivity values used for scaling analysis when calculating the total diagnostic time.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgement

This work was supported by the Toyota Research Institute through D3BATT: Center for Data-Driven Design of Li-Ion Batteries.

## Citation

If you used this code, please cite this Software as:

@article{rhyu2024optimum,
  
  title={Optimum Model-Based Design of Diagnostics Experiments (DOE) with Hybrid Pulse Power Characterization (HPPC) for Lithium-Ion Batteries},
  
  author={Rhyu, Jinwook and Zhuang, Debbie and Bazant, Martin Z and Braatz, Richard D},
  
  journal={Journal of The Electrochemical Society},
  
  volume={171},
  
  number={7},
  
  pages={070544},
  
  year={2024},
  
  publisher={IOP Publishing}
  
}
