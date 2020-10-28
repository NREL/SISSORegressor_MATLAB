Implementation of the SISSO regression algorithm in MATLAB by Paul Gasper. The SISSO regression algorithm iteratively selects model features from candidates, converging even when the number of possible features is much greater than the number of available data points.
Includes a test script that validates the algorithm by replicating the results using data and procedure from https://analytics-toolkit.nomad-coe.eu/hub/user-redirect/notebooks/tutorials/compressed_sensing.ipynb.
Python code from the SISSO regressor in the above Jupyter notebook was used as the basis for developing the MATLAB implementation.
The SISSO regression algorithm is detailed by the original authors in R. Ouyang, S. Curtarolo, E. Ahmetcik et al., Phys. Rev. Mater. 2, 083802 (2018), R. Ouyang, E. Ahmetcik, C. Carbogno, M. Scheffler, and L. M. Ghiringhelli, J. Phys.: Mater. 2, 024002 (2019).
A more efficient Fortran implementation can be found at https://github.com/rouyang2017/SISSO.
