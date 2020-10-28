% Paul Gasper, NREL, September 2020
% Test the use of the Sisso Regressor. Data and generated features have
% been copied from the 'Compressed Sensing' example published online: 
% https://analytics-toolkit.nomad-coe.eu/hub/user-redirect/notebooks/tutorials/compressed_sensing.ipynb
% This Jupyter notebook demonstrates findings published in the paper:
% L. M. Ghiringhelli, J. Vybiral, S. V. Levchenko, C. Draxl, M. Scheffler: 
% Big Data of Materials Science: Critical Role of the Descriptor, Phys. Rev. Lett. 114, 105503 (2015)
% This demonstrates that the SissoRegressor (translated into
% Matlab from Python from code from the same Jupyter notebook) class is
% operating.
clear; clc; close all;

%% Test on 'small' feature set (115 features, 82 data points):
% Load files:
%   feature_list: cell array of chacter vectors
%   X: array of input data, each column is a feature
%   Y: column vector of response data
load('sisso_test_data.mat')
fprintf("Fitting 'small' data: %d data points, %d features.\n",length(Y), size(X,2))

% Expected result for this data (copied from the Jupyter notebook):
%           RMSE            Model
% 1D:	0.296668	- 0.484 (r_p(A)+r_d(B)) + 1.944
% 2D:	0.218070	- 3.483 (r_p(A)+r_d(B)) + 0.392 (r_p(A)+r_d(B))^2 + 7.495
% 3D:	0.193928	- 3.528 (r_p(A)+r_d(B)) + 0.405 (r_p(A)+r_d(B))^2 + 0.293 |r_s(A)-r_d(B)| + 7.280
% Execution time on Jupyter: 406 ms

% Test
n_nonzero_coefs = 3; n_features_per_sis_iter = 10;
fprintf("Searching for models up to %d dimemsions, considering %d new features per iteration.\n",n_nonzero_coefs, n_features_per_sis_iter)
sisso = SissoRegressor(n_nonzero_coefs, n_features_per_sis_iter);
sisso = sisso_fit(sisso, X, Y);
print_models(sisso, feature_list)

% Matlab solution:
% (formatted slightly different cause I like it better)
%           RMSE            Model
% 1D: 	0.296696	1.922 - 0.478 (r_p(A)+r_d(B)) 
% 2D: 	0.218070	7.495 - 3.483 (r_p(A)+r_d(B)) + 0.392 (r_p(A)+r_d(B))^2 
% 3D: 	0.193928	7.280 - 3.528 (r_p(A)+r_d(B)) + 0.405 (r_p(A)+r_d(B))^2 + 0.293 |r_s(A)-r_d(B)| 
% Execution time on Matlab: 116 ms (faster!)
% 1D case is not coming out with the exact same coef/intercept, but the
% RMSE is almost exactly the same, so perhaps the linear solution in that
% case is simply underdetermined.

%% Test on 'large' feature set (3391 features, 82 data points):
% Load files:
%   feature_list: cell array of chacter vectors
%   X: array of input data, each column is a feature
%   Y: column vector of response data
load('sisso_test_data_big.mat')
fprintf("Fitting 'big' data: %d data points, %d features.\n",length(Y), size(X,2))

% Expected result for this data (copied from Jupyter notebook):
%           RMSE            Model
% 1D:	0.137212	- 0.055 (IP(A)+IP(B))/r_p(A)^2 - 0.332
% 2D:	0.100216	+ 0.114 |IP(B)-EA(B)|/r_p(A)^2 - 1.482 |r_s(A)-r_p(B)|/exp(r_s(A)) - 0.145
% 3D:	0.076428	+ 0.109 |IP(B)-EA(B)|/r_p(A)^2 - 1.766 |r_s(A)-r_p(B)|/exp(r_s(A)) - 6.032 |r_s(B)-r_p(B)|/(r_p(B)+r_d(A))^2 - 0.005
% Execution time on Jupyter: 5.16 s

% Test (Note that there are more features per iteration as well as more features):
n_nonzero_coefs = 3; n_features_per_sis_iter = 26;
fprintf("Searching for models up to %d dimemsions, considering %d new features per iteration.\n",n_nonzero_coefs, n_features_per_sis_iter)
sisso_big = SissoRegressor(n_nonzero_coefs, n_features_per_sis_iter);
sisso_big = sisso_fit(sisso_big, X, Y);
print_models(sisso_big, feature_list)

% Matlab solutions:
%           RMSE            Model
% 1D: 	0.137310	-0.327 - 0.055 (IP(A)+IP(B))/r_p(A)^2 
% 2D: 	0.100216	-0.145 + 0.114 |IP(B)-EA(B)|/r_p(A)^2 - 1.482 |r_s(A)-r_p(B)|/exp(r_s(A)) 
% 3D: 	0.076428	-0.005 + 0.109 |IP(B)-EA(B)|/r_p(A)^2 - 1.766 |r_s(A)-r_p(B)|/exp(r_s(A)) - 6.032 |r_s(B)-r_p(B)|/(r_p(B)+r_d(A))^2 
% Execution time on Matlab: 0.593 s (~ 10x faster)