# Lie_transformation

Small Python module to generate the Hamiltonian for a given multipole order, and find the Lie transformation accordingly. 

Possibility to make the mqp symplectic by computing the corresponding Yoshida coefficients. Then compare the results for a closed-form solution, truncated map, and symplectic truncated map.

Outputs are made with ipywidgets and elements strength can be varied interactively, or outputs can be saved as gifs (see below).


### Comparison between the closed-form solution and the truncated map (at order 6) for quadrupole

![](https://github.com/ColasDroin/Lie_transformation/blob/master/gifs/1.gif)


### Comparison between all maps in phase-space and real space for quadrupole


![](https://github.com/ColasDroin/Lie_transformation/blob/master/gifs/3.gif)


### Comparison between all maps in phase-space and real space for non-linear element (sextupole)


![](https://github.com/ColasDroin/Lie_transformation/blob/master/gifs/4.gif)