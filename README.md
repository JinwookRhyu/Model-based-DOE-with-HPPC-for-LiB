This repository contains the software for [Optimum Model-Based Design of Diagnostics Experiments (DOE) with Hybrid Pulse Power Characterization (HPPC) for Lithium-Ion Batteries] which can be used for performing model-based DOE for optimizing the HPPC protocol. This software is associated with the paper 'Optimum Model-Based Design of Diagnostics Experiments (DOE) with Hybrid Pulse Power Characterization (HPPC) for Lithium-Ion Batteries' by Jinwook Rhyu et al.



# Code

The software is performed in Python where design_generalized_optimal_HPPC.py and perform_MCMC.py are the main functions for performing model-based DoE and MCMC simulations, respectively.
MCMC_autocorrelation_help.py is a helper function for performing MCMC simulations.
extract_J1_J2_from_protocol.py is a function for calculating the two objective functions (f_uncertainty and f_time) when the protocol is given.
Figures.py is a function used to generate a figure for MCMC results.

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
  year={2024}
}
