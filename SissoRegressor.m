%{
Paul Gasper, NREL, September 2020

SISSO (Sure-Independence Screening -> Sparsifying Operator (here, L0 regularization)):
R. Ouyang, S. Curtarolo, E. Ahmetcik et al., Phys. Rev. Mater.2, 083802 (2018)
R. Ouyang, E. Ahmetcik, C. Carbogno, M. Scheffler, and L. M. Ghiringhelli, J. Phys.: Mater. 2, 024002 (2019).

Code has been translated into MATLAB from the Python implementation of SISSO, which can be found at:
https://analytics-toolkit.nomad-coe.eu/hub/user-redirect/notebooks/tutorials/compressed_sensing.ipynb
%}

classdef SissoRegressor
    properties
        % Input properties:
        % Number of non-zero coefficients / maximum number of dimension of the model
        n_nonzero_coefs {mustBeNumeric} = 1
        % Number of features collected per SIS step (these are then searched exhaustively for models with 1:n_non_zero_coeffs dimension.
        n_features_per_sis_iter {mustBeNumeric} = 1
        % If true, in the L0 step all combinations of sis_collected features will be checked.
        % If false, combinations of features of the same SIS step will be neglected
        all_L0_combinations {mustBeNumericOrLogical} = true;
        
        % Internal properties:
        coefs = [];
        coefs_stan = [];
        intercept = [];
        list_of_coefs = {};
        list_of_nonzero_coefs = {};
        list_of_intercepts = {};
        rmses = [];
        sis_selected_indices = {};
        sis_not_selected_indices = [];
        L0_selected_indices = {};
        curr_selected_indices = [];
        scales = 1;
        means = 0;
    end
    methods
        function obj = SissoRegressor(n_nonzero_coefs, n_features_per_sis_iter, all_L0_combinations)
            if nargin == 0
                obj.n_nonzero_coefs = 1;
                obj.n_features_per_sis_iter = 1;
                obj.all_L0_combinations = true;
            elseif nargin == 1
                obj.n_nonzero_coefs = n_nonzero_coefs;
                obj.n_features_per_sis_iter = 1;
                obj.all_L0_combinations = true;
            elseif nargin == 2
                obj.n_nonzero_coefs = n_nonzero_coefs;
                obj.n_features_per_sis_iter = n_features_per_sis_iter;
                obj.all_L0_combinations = true;
            elseif nargin == 3
                obj.n_nonzero_coefs = n_nonzero_coefs;
                obj.n_features_per_sis_iter = n_features_per_sis_iter;
                obj.all_L0_combinations = all_L0_combinations;
            end
%             obj.n_nonzero_coefs = n_nonzero_coefs;
%             obj.n_features_per_sis_iter = n_features_per_sis_iter;
%             obj.all_L0_combinations = all_L0_combinations;
        end
        
        function obj = sisso_fit(obj, X, Y)
            %obj = SISSO_FIT(obj, X, Y)
            % Runs the regression algorithm, using the n_nonzero_coeffs and
            % n_features_per_sis_iter properties of obj.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   X (double): Array of values of potential model features.
            %   Y (double): Vector of the response variable values.
            % Output:
            %   obj: Instance of the SissoRegressor class, with the results
            %   from the regression stored in the class properties.
            
            % Turn off warnings, there are often rank-dificient matrices
            % when running the L0 optimization at higher dimensions.
            warning('off')
            % Use the sisso regressor to optimize the
            % X: matrix of input features for each response, [N_samples, N_features]
            % Y: vector of response data, [N_sample,1]
            check_params(obj,size(X,2));
            obj.sis_not_selected_indices = 1:length(X);
            % Standardize input features:
            obj = set_standardizer(obj, X);
            X = get_standardized(obj, X);
            % Center response variable:
            Y_mean = mean(Y);
            Y_centered = Y - Y_mean;
            
            for iter = 1:obj.n_nonzero_coefs
                % Residuals for the first iteration are just the centered P
                % value (model is simply the mean)
                Residual = [];
                if iter == 1
                    Residual = Y_centered;
                else
                    Residual = Y_centered - predict_of_standardized(obj,X);
                end
                
                % SIS: get the indicies of the n_features_per_sis_iter that
                % are closest to the residuals (first iter -> response,
                % following iters, residuals of the model from the previous
                % iter)
                [obj, indices_n_closest, best_projection_score] = sis(obj, X, Residual);
                obj.sis_selected_indices = [obj.sis_selected_indices indices_n_closest];
                
                % SA step or L0 step (if iter > 1)
                if iter == 1
                    % RMSE of response - intercept
                    obj.coefs_stan = best_projection_score/length(Y);
                    obj.curr_selected_indices = indices_n_closest(1);
                    rmse = sqrt(sum((Y_centered - predict_of_standardized(obj,X)).^2) ./ length(Y));
                else
                    % Perform L0 regularization
                    [obj.coefs_stan, obj.curr_selected_indices, rmse] = L0_regularization(obj, X, Y_centered, obj.sis_selected_indices);
                end
                
                % Process and save model outcomes
                % Transform standardized coefs into original scale coefs
                [coefs_temp, obj.intercept] = get_notstandardized_coefs(obj, obj.coefs_stan, Y_mean);
                % Generate coefs array with zeros except the selected indices
                obj.coefs = zeros(1,size(X,2));
                obj.coefs(obj.curr_selected_indices) = coefs_temp;
                % Append lists of coefs, indicies, rmses...
                obj.list_of_nonzero_coefs = [obj.list_of_nonzero_coefs coefs_temp];
                obj.list_of_coefs = [obj.list_of_coefs obj.coefs];
                obj.list_of_intercepts = [obj.list_of_intercepts obj.intercept];
                obj.L0_selected_indices = [obj.L0_selected_indices obj.curr_selected_indices];
                obj.rmses = [obj.rmses rmse];
            end
        end
        
        function Y_pred = sisso_predict(obj, X, dim)
            %Y_pred = SISSO_PREDICT(obj, X, dim)
            % Predict the response for the given input data from the using
            % the previous sisso model (or the one specified by 'dim').
            % Inputs:
            %   X (double): array of input features for each response, 
            %     [N_samples, N_features]
            %   dim (int): index of desired fitted SISSO model
            % Output:
            %   Y_pred (double): Vector of predicted response variable
            %   values.
            
            if isempty(dim)
                dim = obj.n_nonzero_coefs;
            end
            
            % Use only selected indices/features of D and add a column of
            % ones for the intercept/bias
            D_model = X(:, obj.L0_selected_indices{dim});
            D_model = [ones(size(X,1),1), D_model];
            coefs_model = [obj.list_of_intercepts{dim}; obj.list_of_nonzero_coefs{dim}];
            Y_pred = D_model * coefs_model;
        end
        
        function print_models(obj, features)
            %PRINT_MODELS(obj, features)
            % Prints the model constructed from the selected feature list.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   features (cell): Cellstr of feature labels.
            fprintf("%14s %16s\n", 'RMSE', 'Model')
            for model_dim = 1:obj.n_nonzero_coefs
                disp_str = get_model_string(obj, features, model_dim);
                disp(disp_str)
            end
            disp(" ");
        end
        
        function check_params(obj, n_columns)
            %CHECK_PARAMS(obj, n_columns)
            % Checks settings of the fit.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   n_columns (int): Number of columns in the X array.
            
            error_str = strcat("n_nonzero_coefs * n_features_per_sis_iter is larger /n"...
                ,"than the number of columns in your input matrix./n"...
                ,"Choose a smaller n_nonzero_coefs or a smaller n_features_per_sis_iter.");
            if n_columns < (obj.n_nonzero_coefs * obj.n_features_per_sis_iter)
                error(error_str)
            end
            % Calculate and display the number of L0 optimizations during
            % the SO step of SISSO:
            n_L0_steps = 0;
            fprintf("L0 optimizations for 1D model: 0\n");
            if obj.all_L0_combinations
                for dim = 2:obj.n_nonzero_coefs
                    L0_calcs_this_iter = nchoosek(obj.n_features_per_sis_iter * dim, dim);
                    n_L0_steps = n_L0_steps + L0_calcs_this_iter;
                    fprintf("L0 optimizations for %dD model: %d\n", dim, L0_calcs_this_iter);
                end
            else
                for dim = 2:obj.n_nonzero_coefs
                    L0_calcs_this_iter = prod(obj.n_features_per_sis_iter * dim);
                    n_L0_steps = n_L0_steps + L0_calcs_this_iter;
                    fprintf("L0 optimizations for %dD model: %d\n", dim, L0_calcs_this_iter);
                end
            end
            fprintf("Total # of L0 optimizations: %d\n", n_L0_steps)
            disp(" ");
        end
        
        function obj = set_standardizer(obj, X)
            %obj = SET_STANDARDIZER(obj, X)
            % Stores the mean and standard deviation of X in the
            % corresponding properties of obj.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   X (double): Array of values of potential model features.
            obj.means = mean(X);
            obj.scales = std(X);
        end
        
        function X = get_standardized(obj, X)
            %obj = GET_STANDARDIZED(obj, X)
            % Standardizes all columns of X.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   X (double): Array of values of potential model features.
            X = (X - obj.means) ./ obj.scales;
        end
        
        function Y_predict = predict_of_standardized(obj, X)
            %Y_predict = PREDICT_OF_STANDARDIZED(obj, X)
            % Calculates the predicition of the response variable using the
            % optimized coefficient values and chosen features from the
            % most recent iteration of sisso_fit using a standardized input
            % feature array X.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   X (double): Array of standardized values of potential model 
            %     features. 
            % Outputs:
            %   Y_predict: Vector of predicted response variable values.
            Y_predict = X(:,obj.curr_selected_indices) * obj.coefs_stan;
        end
        
        function [obj, indices_n_closest_out, best_projection_score] = sis(obj, X, Y)
            %[obj, indices_n_closest_out, best_projection_score] = SIS(obj, X, Y)
            % Finds the n_features_per_sis_iter feature columns with the
            % lowest projection scores compared to 1) Y (if this is the
            % first iteration of SIS) or 2) the residual errors of the
            % previous SISSO iteration.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   X (double): Array of values of potential model features.
            %   Y (double): Vector of the values of the response variable.
            % Outputs:
            %   obj: Instance of the SissoRegressor class, with some
            %     regression results stored in obj properties.
            %   indices_n_closest_out (int): Column indicies of the
            %     features selected during this SIS iteration.
            %   best_projection_score (double): The best projection score
            %     from this SIS iteration.
            
            % Evaluate how close each feature is to the target without
            % already selected indicies from prior iterations
            projection_scores = Y'*X(:,obj.sis_not_selected_indices);
            abs_projection_scores = abs(projection_scores);
            
            % Sort the values according to their absolute projection score
            % starting from the closest, and get the indices of the
            % n_feature_per_sis_iter closest values.
            [~,indices_sorted] = sort(abs_projection_scores);
            indices_sorted = flip(indices_sorted);
            indices_n_closest = indices_sorted(1:obj.n_features_per_sis_iter);
            best_projection_score = projection_scores(indices_n_closest(1));
            
            % Transform indices_n_closest according to original indices of
            % 1:size(D,2) and delete the selected ones from
            % obj.sis_not_slected_indices
            indices_n_closest_out = obj.sis_not_selected_indices(indices_n_closest);
            obj.sis_not_selected_indices = obj.sis_not_selected_indices(~any(obj.sis_not_selected_indices == indices_n_closest(:)));
        end
        
        function [coefs_stan, curr_selected_indices, rmse] = L0_regularization(obj, X, Y, list_of_sis_indices)
            %[coefs_stan, curr_selected_indices, rmse] = L0_REGULARIZATION(obj, X, Y, list_of_sis_indices)
            % Exhaustively searches models formed by all possible
            % combinations of the features in list_of_sis_indices to find
            % the best model structure (curr_selected_indicies),
            % coefficients (coefs_stan), and root mean square error (rmse).
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   X (double): Array of values of potential model features.
            %   Y (double): Vector of the values of the response variable.
            %   list_of_sis_indices (cell): cell array with arrays of the
            %     indices of possible features output from each iteration
            %     of the SIS step.
            % Outputs:
            %   coefs_stan (double): Array of coefficient values for the
            %     best model.
            %   curr_selected_indices (int): Array of indices denoting
            %     which feature columns are used by the best model.
            %   rmse (double): Root mean square error of the prediction
            %     made by the best model.
            
            square_error_min = Y'*Y;
            coefs_min = [];
            indices_combi_min = [];
            
            % Check each least sqaures error of combination of each indices
            % array of list_of_sis_indices. If obj.all_L0_combinations is
            % false, combinations of features from the same SIS iteration
            % will be neglected.
            if obj.all_L0_combinations
                combinations_generator = combnk([list_of_sis_indices{:}], length(list_of_sis_indices));
            else
                combinations_generator = cart_prod(list_of_sis_indices);
            end
            
            for i = 1:size(combinations_generator,1)
                indices_combi = combinations_generator(i,:);
                D_Ls = X(:,indices_combi);
                coefs_Ls = D_Ls\Y;
                residuals = Y - (D_Ls * coefs_Ls);
                square_error = sum(residuals.^2);
                %try
                %%%% I believe this 'try' is here because in Python, the
                %%%% linear eq. solver can return an empty var for the
                %%%% square error, and trying to compare an empty var may
                %%%% result in error. Here, the square error cannot be
                %%%% empty.
                if square_error < square_error_min
                    square_error_min = square_error;
                    coefs_min = coefs_Ls;
                    indices_combi_min = indices_combi;
                end
                %catch
                    %%%% Pass
                %end
            end
            coefs_stan = coefs_min;
            curr_selected_indices = indices_combi_min;
            rmse = sqrt(square_error_min / size(X,1));
        end
        
        function [coefs_orig, intercept_orig] = get_notstandardized_coefs(obj, coefs_stan, intercept_stan)
            %[coefs_orig, intercept_orig] = GET_NONSTANDARDIZED_COEFS(obj, coefs_stan, intercept_stan)
            % Transform coefs of a linear model with standardized input to
            % coefs of a linear model with original (nonstandardized) input.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   coefs_stan (double): Coefficient values assuming
            %     standardized input features.
            %   intercept_stan (double): Model intercept value assuming 
            %     standardized input features.
            % Outputs:
            %   coefs_orig (double): Coefficient values for raw input
            %     features.
            %   intercept_orig (double): Model intercept value for original
            %     input features.
            coefs_orig = coefs_stan ./ obj.scales(obj.curr_selected_indices)';
            intercept_orig = intercept_stan - (obj.means(obj.curr_selected_indices) ./ obj.scales(obj.curr_selected_indices))*coefs_stan;
        end
        
        function model_str = get_model_string(obj, features, model_dim)
            %model_str = GET_MODEL_STRING(obj, features, model_dim)
            % Creates a nice string for displaying potential models found
            % during the operation of sisso_fit.
            % Inputs:
            %   obj: Instance of the SissoRegressor class.
            %   features (cell): Cellstr of feature labels.
            %   model_dim (int): Dimension of the model stored in the
            %     properties of obj to create a string for.
            % Output:
            %   model_str (char): Nice string displaying the resulting
            %     model of dimension model_dim.
            
            coefs_dim = [obj.list_of_intercepts{model_dim}, obj.list_of_nonzero_coefs{model_dim}'];
            selected_features = {[]};
            for idx = obj.L0_selected_indices{model_dim}
                selected_features = [selected_features features(idx)];
            end
            model_str = sprintf("%dD: \t%8f\t", model_dim, obj.rmses(model_dim));
            for dim = 1:model_dim+1
                c = coefs_dim(dim);
                c_abs = abs(c);
                if c > 0
                    sign = '+';
                else
                    sign = '-';
                end
                if dim == 1 % intercept
                    model_str = strcat(model_str, sprintf("%0.3f ", c));
                else % feature
                    model_str = strcat(model_str, sprintf("%s %0.3f %s ", sign, c_abs, selected_features{dim}));
                end
            end
        end
        
        function X = cart_prod(sets)
            %X = CART_PROD(sets)
            % Modified from https://www.mathworks.com/matlabcentral/fileexchange/5475-cartprod-cartesian-product-of-multiple-sets
            % This is to copy the behavior of itertools.product in Python.
            
            %CARTPROD Cartesian product of multiple sets.
            %
            %   X = CARTPROD(A,B,C,...) returns the cartesian product of the sets
            %   A,B,C, etc, where A,B,C, are numerical vectors.
            %
            %   Example: A = [-1 -3 -5];   B = [10 11];   C = [0 1];
            %
            %   X = cartprod(A,B,C)
            %   X =
            %
            %     -5    10     0
            %     -3    10     0
            %     -1    10     0
            %     -5    11     0
            %     -3    11     0
            %     -1    11     0
            %     -5    10     1
            %     -3    10     1
            %     -1    10     1
            %     -5    11     1
            %     -3    11     1
            %     -1    11     1
            %
            %   This function requires IND2SUBVECT, also available on the MathWorks
            %   File Exchange site.
            
            numSets = length(sets);
            for i = 1:numSets
                % Check each cell entry, sort it if its okay.
                thisSet = sort(sets{i});
                if ~isequal(numel(thisSet),length(thisSet))
                    error('All inputs must be vectors.')
                end
                if ~isnumeric(thisSet)
                    error('All inputs must be numeric.')
                end
                if ~isequal(thisSet,unique(thisSet))
                    error(['Input set' ' ' num2str(i) ' ' 'contains duplicated elements.'])
                end
                sizeThisSet(i) = length(thisSet);
                sets{i} = thisSet;
            end
            X = zeros(prod(sizeThisSet),numSets);
            for i = 1:size(X,1)
                % Envision imaginary n-d array with dimension "sizeThisSet" ...
                % = length(varargin{1}) x length(varargin{2}) x ...
                ixVect = ind2subVect(sizeThisSet,i);
                for j = 1:numSets
                    X(i,j) = sets{j}(ixVect(j));
                end
            end
        end
        
        function X = ind2subVect(siz, ndx)
            %X = IND2SUBVECT(siz, ndx)
            % From https://www.mathworks.com/matlabcentral/fileexchange/5476-ind2subvect-multiple-subscript-vector-from-linear-index
            
            %IND2SUBVECT Multiple subscripts from linear index.
            %   IND2SUBVECT is used to determine the equivalent subscript values
            %   corresponding to a given single index into an array.
            %
            %   X = IND2SUBVECT(SIZ,IND) returns the matrix X = [I J] containing the
            %   equivalent row and column subscripts corresponding to the index
            %   matrix IND for a matrix of size SIZ.
            %
            %   For N-D arrays, X = IND2SUBVECT(SIZ,IND) returns matrix X = [I J K ...]
            %   containing the equivalent N-D array subscripts equivalent to IND for
            %   an array of size SIZ.
            %
            %   See also IND2SUB.  (IND2SUBVECT makes a one-line change to IND2SUB so as
            %   to return a vector of N indices rather than retuning N individual
            %   variables.)%IND2SUBVECT Multiple subscripts from linear index.
            %   IND2SUBVECT is used to determine the equivalent subscript values
            %   corresponding to a given single index into an array.
            %
            %   X = IND2SUBVECT(SIZ,IND) returns the matrix X = [I J] containing the
            %   equivalent row and column subscripts corresponding to the index
            %   matrix IND for a matrix of size SIZ.
            %
            %   For N-D arrays, X = IND2SUBVECT(SIZ,IND) returns matrix X = [I J K ...]
            %   containing the equivalent N-D array subscripts equivalent to IND for
            %   an array of size SIZ.
            %
            %   See also IND2SUB.  (IND2SUBVECT makes a one-line change to IND2SUB so as
            %   to return a vector of N indices rather than returning N individual
            %   variables.)
            
            % All MathWorks' code from IND2SUB, except as noted:
            n = length(siz);
            k = [1 cumprod(siz(1:end-1))];
            ndx = ndx - 1;
            for i = n:-1:1
                X(i) = floor(ndx/k(i))+1;      % replaced "varargout{i}" with "X(i)"
                ndx = rem(ndx,k(i));
            end
        end
    end
end