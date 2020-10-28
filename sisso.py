# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
Init signature:
SissoRegressor(
    n_nonzero_coefs=1,
    n_features_per_sis_iter=1,
    all_l0_combinations=True,
)
Docstring:     
A simple implementation of the SISSO algorithm (R. Ouyang, S. Curtarolo, 
E. Ahmetcik et al., Phys. Rev. Mater.2, 083802 (2018)) for regression for
workshop tutorials. SISSO is an iterative approach where at each iteration
first SIS (Sure Independence Sreening) and SO (Sparsifying Operator, here l0-regularization) 
is applied.
Note that it becomes the orthogonal matching pursuit for n_features_per_sis_iter=1.

This code was copied from:
https://analytics-toolkit.nomad-coe.eu/hub/user-redirect/notebooks/tutorials/compressed_sensing.ipynb
A more efficient fortran implementation can be found on https://github.com/rouyang2017.

The code raises an error for parameters that lead to longer calculation in order
to keep the tutorial servers free.

Parameters
----------
n_nonzero_coefs : int
    Number of nonzero coefficients/ max. number of dimension of descriptor.

n_features_per_sis_iter : int
    Number of features collected per SIS step.

all_l0_combinations : bool, default True
    If True, in the l0 step all combinations out sis_collected features will be checked.
    If False, combinations of features of the same SIS iterations will be neglected.

Attributes
----------
coefs: array, [n_features]
    (Sparse) coefficient vector of linear model

intercept: int
    Intercept/ bias of linear model.

sis_selected_indices : list of arrays, [[n_features_per_sis_iter,], [n_features_per_sis_iter,], ...]
    List of indices selected at each SIS iteration.

l0_selected_indices : list of arrays, [[1,], [2,], ...]
    List of indices selected at each SIS+L0 iteration.

Methods
-------
fit(D, P) : P: array, [n_sample, n_features]
            D: array, [n_sample,]
    
predict(D[, dim]): D: array, [n_sample,]
                   dim: int, optional
                       dim (number of nonzero coefs) specifies that prediction 
                       should be done with result from another step than the last.

print_models(features) : features: list of str [n_features,]
Source:        
"""
class SissoRegressor(object):
    """ A simple implementation of the SISSO algorithm (R. Ouyang, S. Curtarolo, 
    E. Ahmetcik et al., Phys. Rev. Mater.2, 083802 (2018)) for regression for
    workshop tutorials. SISSO is an iterative approach where at each iteration
    first SIS (Sure Independence Sreening) and SO (Sparsifying Operator, here l0-regularization) 
    is applied.
    Note that it becomes the orthogonal matching pursuit for n_features_per_sis_iter=1.
    
    A more efficient fortran implementation can be found on https://github.com/rouyang2017.

    The code raises an error for parameters that lead to longer calculation in order
    to keep the tutorial servers free.
    
    Parameters
    ----------
    n_nonzero_coefs : int
        Number of nonzero coefficients/ max. number of dimension of descriptor.

    n_features_per_sis_iter : int
        Number of features collected per SIS step.

    all_l0_combinations : bool, default True
        If True, in the l0 step all combinations out sis_collected features will be checked.
        If False, combinations of features of the same SIS iterations will be neglected.

    Attributes
    ----------
    coefs: array, [n_features]
        (Sparse) coefficient vector of linear model

    intercept: int
        Intercept/ bias of linear model.
    
    sis_selected_indices : list of arrays, [[n_features_per_sis_iter,], [n_features_per_sis_iter,], ...]
        List of indices selected at each SIS iteration.

    l0_selected_indices : list of arrays, [[1,], [2,], ...]
        List of indices selected at each SIS+L0 iteration.

    Methods
    -------
    fit(D, P) : P: array, [n_sample, n_features]
                D: array, [n_sample,]
        
    predict(D[, dim]): D: array, [n_sample,]
                       dim: int, optional
                           dim (number of nonzero coefs) specifies that prediction 
                           should be done with result from another step than the last.
    
    print_models(features) : features: list of str [n_features,]
    """
    
    def __init__(self, n_nonzero_coefs=1, n_features_per_sis_iter=1, all_l0_combinations=True):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_features_per_sis_iter = n_features_per_sis_iter
        self.all_l0_combinations = all_l0_combinations
    
    def fit(self, D, P):
        self._check_params(D.shape[1])
        self._initialize_variables()

        self.sis_not_selected_indices = np.arange(D.shape[1])
        # standardize D and center P
        self._set_standardizer(D)
        D = self._get_standardized(D)
        
        # center target
        P_mean = P.mean()
        P_centered = P - P.mean()
        
        for i_iter in range(self.n_nonzero_coefs):
            # set residual, in first iteration the target P is used
            if i_iter == 0:
                Residual = P_centered
            else:
                Residual = P_centered - self._predict_of_standardized(D)
            
            # sis step: get the indices of the n_features_per_sis_iter
            # closest features to the residual/target
            indices_n_closest, best_projection_score = self._sis(D, Residual)
            self.sis_selected_indices.append( indices_n_closest )
            
            # SA step or L0 step, only if i_iter > 0
            if i_iter == 0:
                self._coefs_stan, self.curr_selected_indices = best_projection_score/P.size, indices_n_closest[:1]
                rmse = np.linalg.norm(P_centered - self._predict_of_standardized(D)) / np.sqrt(P.size)
            else:
                # perform L0 regularization
                self._coefs_stan, self.curr_selected_indices, rmse = self._l0_regularization(D, P_centered, self.sis_selected_indices)
            
            ### process and save model outcomes
            # transform coefs to coefs of not standardized D
            coefs, self.intercept = self._get_notstandardized_coefs(self._coefs_stan, P_mean)
            
            # generate coefs array with zeros except selected indices 
            # (might be needed for the user)
            self.coefs = np.zeros(D.shape[1])
            self.coefs[self.curr_selected_indices] = coefs
            
            # append lists of coefs, indices, ...
            self._list_of_coefs.append(coefs)
            self.list_of_coefs.append(self.coefs)
            self.list_of_intercepts.append(self.intercept)
            self.l0_selected_indices.append(self.curr_selected_indices)
            self.rmses.append(rmse)
    
    def predict(self, D, dim=None):
        if dim is None:
            dim = self.n_nonzero_coefs
        
        # use only selected indices/features of D
        # and add column of ones for the intercept/bias
        D_model = D[:, self.l0_selected_indices[dim - 1]]
        D_model = np.column_stack((D_model, np.ones(D.shape[0])))
        coefs_model = np.append(self._list_of_coefs[dim - 1], self.list_of_intercepts[dim - 1])

        return np.dot(D_model, coefs_model)
     
    def print_models(self, features):
        string = '%14s %16s\n' %('RMSE', 'Model')
        string += "\n".join( [self._get_model_string(features, i_iter) for i_iter in range(self.n_nonzero_coefs)] )
        print(string)
    
    def _initialize_variables(self):
        # variabels for standardizer
        self.scales = 1.
        self.means = 0.

        # indices selected SIS 
        self.sis_selected_indices = []
        self.sis_not_selected_indices = None
        
        # indices selected by L0 (after each SIS step)
        self.l0_selected_indices = []
        self.curr_selected_indices = None
        
        # coefs and lists for output
        self.coefs = None
        self.intercept = None
        self.list_of_coefs = []
        self._list_of_coefs = []
        self.list_of_intercepts = []
        self.rmses = []

    def _set_standardizer(self, D):
        self.means  = D.mean(axis=0)
        self.scales = D.std(axis=0)
    
    def _get_standardized(self, D):
        return (D - self.means) / self.scales

    def _l0_regularization(self, D, P, list_of_sis_indices):
        square_error_min = np.inner(P, P)
        coefs_min, indices_combi_min = None, None

        # check each least squares error of combination of each indeces tuple of list_of_sis_indices.
        # If self.all_l0_combinations is False, combinations of featuers from the same SIS iteration
        # will be neglected
        if self.all_l0_combinations:
            combinations_generator = combinations(np.concatenate(list_of_sis_indices), len(list_of_sis_indices))
        else:
            combinations_generator = product(*list_of_sis_indices)

        for indices_combi in combinations_generator:
            D_ls = D[:, indices_combi]
            coefs, square_error, __1, __2 = np.linalg.lstsq(D_ls, P, rcond=-1)
            try:
                if square_error[0] < square_error_min: 
                    square_error_min = square_error[0]
                    coefs_min, indices_combi_min = coefs, indices_combi
            except:
                pass
        rmse = np.sqrt(square_error_min / D.shape[0])
        return coefs_min, list(indices_combi_min), rmse

    def _get_notstandardized_coefs(self, coefs, bias):
        """ transform coefs of linear model with standardized input to coefs of non-standardized input"""
        coefs_not = coefs / self.scales[self.curr_selected_indices] 
        bias_not  = bias - np.dot(self.means[self.curr_selected_indices] / self.scales[self.curr_selected_indices], coefs)
        
        return coefs_not, bias_not

    def _ncr(self, n, r):
        """ Binomial coefficient"""
        r = min(r, n-r) 
        if r == 0: return 1
        numer = functools.reduce(op.mul, range(n, n-r, -1)) 
        denom = functools.reduce(op.mul, range(1, r+1))
        return numer//denom

    def _check_params(self, n_columns):
        string_out  = "n_nonzero_coefs * n_features_per_sis_iter is larger " 
        string_out += "than the number of columns in your input matrix. "
        string_out += "Choose a smaller n_nonzero_coefs or a smaller n_features_per_sis_iter."
        if n_columns < self.n_nonzero_coefs * self.n_features_per_sis_iter:
            raise ValueError(string_out)

        # Shrinkage sisso for tutorials in order to save tutorial server
        # cpus from beeing occupied all the time
        #n_l0_steps = sum([self.ncr( n_sis*dim, dim )  for dim in range(1, n_nonzero_coefs+1)])
        
        if self.all_l0_combinations:
            n_l0_steps = sum([self._ncr(self.n_features_per_sis_iter * dim, dim )  for dim in range(2, self.n_nonzero_coefs+1)])
        else:
            n_l0_steps = sum([np.product([self.n_features_per_sis_iter]*dim)  for dim in range(2, self.n_nonzero_coefs+1)])
        
        upper_limit = 80000
        if n_l0_steps > upper_limit:
            string_out  = "With the given settings in the l0-regularizaton %s combinations of features have to be considered." % n_l0_steps
            string_out += "For this tutorial the upper limit is %s. " % upper_limit
            string_out += "Choose a smaller n_nonzero_coefs or a smaller n_features_per_sis_iter."
            raise ValueError(string_out)
            
    def _predict_of_standardized(self, D):
        return np.dot(D[:, self.curr_selected_indices], self._coefs_stan)
       
    def _sis(self, D, P):

        # evaluate how close each feature is to the target 
        # without already selected self.sis_selected_indices
        projection_scores = np.dot(P, D[:, self.sis_not_selected_indices])
        abs_projection_scores = abs(projection_scores)
        
        # sort the values according to their abs. projection score
        # starting from the closest, and get the indices
        # of the n_features_per_sis_iter closest
        indices_sorted = abs_projection_scores.argsort()[::-1]
        indices_n_closest = indices_sorted[: self.n_features_per_sis_iter]
        best_projection_score = projection_scores[ indices_n_closest[:1] ]
        
        # transform indices_n_closest according to originial indices system 
        # of range(D.shape[1]) and delete the selected ones from
        # self.sis_not_selected_indices
        indices_n_closest_out = self.sis_not_selected_indices[indices_n_closest]
        self.sis_not_selected_indices = np.delete(self.sis_not_selected_indices, indices_n_closest)
        
        return indices_n_closest_out, best_projection_score

    def _get_model_string(self, features, i_iter):
        dimension = i_iter + 1
        coefs = np.append(self._list_of_coefs[i_iter], self.list_of_intercepts[i_iter])
        selected_features = [features[i] for i in self.l0_selected_indices[i_iter]]

        string = '%sD:\t%8f\t' %(dimension, self.rmses[i_iter])
        for i_dim in range(dimension+1):
            if coefs[i_dim] > 0.:
                sign = '+' 
                c = coefs[i_dim]
            else:
                sign = '-'
                c = abs(coefs[i_dim]) 
            if i_dim < dimension:
                string += '%s %.3f %s ' %(sign, c, selected_features[i_dim])
            else:
                string += '%s %.3f' %(sign, c)
        return string
"""
File:           /opt/conda/lib/python3.7/site-packages/compressed_sensing/sisso.py
Type:           type
Subclasses:     
"""