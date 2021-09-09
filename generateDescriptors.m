function XOut = generateDescriptors(x, xVars)
% XOut = GENERATEDESCRIPTORS(x, xVars)
% GENERATEDESCRIPTORS uses an algorithmic approach to generate new
% descriptors from a set of input features. Simple operators are applied to
% input features and generated descriptors in order to generate a set of
% descriptors that may better capture non-linear relationships or
% interactions between features. The order that operators are applied, or
% the number of different interactions that are considered, is tailored by
% modifying the code below.
% Inputs:
%   x (cell): cell array of input features, where each cell is a matrix of
%     features with the same units. For instance, consider we have two
%     features with units in Kelvin (T1, T2), three features with units in
%     Volts (V1, V2, V3), and one feature with units in meters (L1). The
%     input x would then be:
%       {[T1, T2], [V1, V2, V3], [L1]}
%     Separating features with different units into cells ensures that we
%     can respect mathematical relationships, i.e., only adding features
%     together if they share the same units (cannot add Volts to Kelvin).
%   xVars (cell): variable names for the input features, ex.:
%       {{'T1', 'T2'}, {'V1', 'V2', 'V3'}, {'L1'}}
% Output:
%   XOut (table): A data table with the input features and all generated
%     descriptors, with each variable automatically named according to the
%     operations used to generate the each descriptor.

% 1st: input features are modified: fourth-root, cube-root, sqrt, squared,
% cubed, fourth-power, inverted.
for idxGroup = 1:length(x)
    % Powers:
    [A1_X,A1_Xvars] = operatorA1(x{idxGroup},xVars{idxGroup});
    [A2_X,A2_Xvars] = operatorA2(x{idxGroup},xVars{idxGroup});
    [A3_X,A3_Xvars] = operatorA3(x{idxGroup},xVars{idxGroup});
    [A4_X,A4_Xvars] = operatorA4(x{idxGroup},xVars{idxGroup});
    [A5_X,A5_Xvars] = operatorA5(x{idxGroup},xVars{idxGroup});
    [A6_X,A6_Xvars] = operatorA6(x{idxGroup},xVars{idxGroup});
    x{idxGroup} = [x{idxGroup},A1_X,A2_X,A3_X,A4_X,A5_X,A6_X];
    xVars{idxGroup} = [xVars{idxGroup},A1_Xvars,A2_Xvars,A3_Xvars,A4_Xvars,A5_Xvars,A6_Xvars];
    % Inverse:
    [B1_X,B1_Xvars] = operatorB1(x{idxGroup},xVars{idxGroup});
    x{idxGroup} = [x{idxGroup},B1_X];
    xVars{idxGroup} = [xVars{idxGroup},B1_Xvars];
end

% 2nd: take the absolute difference of all features with like units, if
% there's more than one feature in the group.
for idxGroup = 1:length(x)
    if size(x{idxGroup}, 2) > 1
        [C2_X, C2_Xvars] = operatorC2(x{idxGroup}, xVars{idxGroup});
        x{idxGroup} = [x{idxGroup}, C2_X];
        xVars{idxGroup} = [xVars{idxGroup}, C2_Xvars];
    end
end

% 3rd: all descriptors of different units are multiplied by one another,
% generating first-order interaction terms.
combinations = nchoosek(1:size(x,2), 2);
for idxCombination = 1:size(combinations, 1)
    thisCombination = combinations(idxCombination, :);
    [x{end+1},xVars{end+1}] = operatorC1(x{thisCombination(1)},x{thisCombination(2)},...
        xVars{thisCombination(1)},xVars{thisCombination(2)});
end

% Unwrap the groups of X and Xvars and combine to get Xout and Xoutvars
x = [x{:}];
xVars = [xVars{:}];
% Remove any columns with infinities. This could be for a number of
% reasons. The most obvious is that a feature can equal 0, and 1/0 = Inf.
% Also, inputs with high powers, or exponentials, can run away sometimes.
Xout = x(:,all(isfinite(x),1));
Xoutvars = xVars(all(isfinite(x),1));
% Assemble the output.
XOut = array2table(Xout, 'VariableNames', Xoutvars);

    function [Xout, Xoutvars] = operatorA1(X, Xvars)
        % Take the sqrt of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^0.5;
            Xoutvars{i} = strcat('(',Xvars{i},'^0.5',')');
        end
    end

    function [Xout, Xoutvars] = operatorA2(X, Xvars)
        % Take the square of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^2;
            Xoutvars{i} = strcat('(',Xvars{i},'^2',')');
        end
    end

    function [Xout, Xoutvars] = operatorA3(X, Xvars)
        % Take the cube of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^3;
            Xoutvars{i} = strcat('(',Xvars{i},'^3',')');
        end
    end

    function [Xout, Xoutvars] = operatorA4(X, Xvars)
        % Take the cube-root of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^(1/3);
            Xoutvars{i} = strcat('(',Xvars{i},'^(1/3)',')');
        end
    end

    function [Xout, Xoutvars] = operatorA5(X, Xvars)
        % Take the fourth-root of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^(1/4);
            Xoutvars{i} = strcat('(',Xvars{i},'^(1/4)',')');
        end
    end

    function [Xout, Xoutvars] = operatorA6(X, Xvars)
        % Take the fourth power of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^4;
            Xoutvars{i} = strcat('(',Xvars{i},'^4',')');
        end
    end

    function [Xout, Xoutvars] = operatorA7(X, Xvars)
        % Take the fifth-root of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^(1/5);
            Xoutvars{i} = strcat('(',Xvars{i},'^(1/5)',')');
        end
    end

    function [Xout, Xoutvars] = operatorA8(X, Xvars)
        % Take the fifth power of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^5;
            Xoutvars{i} = strcat('(',Xvars{i},'^5',')');
        end
    end

    function [Xout, Xoutvars] = operatorB1(X, Xvars)
        % Get the inverse of each feature.
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = 1./X(:,i);
            Xoutvars{i} = strcat('(1/',Xvars{i},')');
        end
    end

    function [Xout, Xoutvars] = operatorB2(X, Xvars)
        % Get the exponential of each feature. ONLY USE FOR NON-DIMENSIONAL
        % FEATURES (parameter coefficient cannot go inside exponential, so
        % the units cannot be converted into the response variable units).
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = exp(X(:,i));
            Xoutvars{i} = strcat('exp(',Xvars{i},')');
        end
    end

    function [Xout, Xoutvars] = operatorC1(X1, X2, X1vars, X2vars)
        % Multiply all features by each other (not including by themselves)
        % Input:
        %   X1: array of features with same units - each column is a feature
        %   X2: array of features with different units than X1
        %   X1vars: cell array of X1 feature var names
        %   X2vars: cell array of X2 feature var names
        % Output formatted similarly to input. Total number of output features is
        %   N_X1*N_X2.
        numdatapoints = size(X1,1);
        numfeaturevars = size(X1,2)*size(X2,2);
        Xout = zeros(numdatapoints,numfeaturevars);
        Xoutvars = cell(1,numfeaturevars);
        idxXout = 1;
        for i = 1:size(X1,2)
            for j = 1:size(X2,2)
                Xout(:,idxXout) = X1(:,i).*X2(:,j);
                Xoutvars{idxXout} = strcat(X1vars{i},'*',X2vars{j});
                idxXout = idxXout + 1;
            end
        end
    end

    function [Xout, Xoutvars] = operatorC2(X1, X1vars)
        % Absolute difference for features of like units
        % Input:
        %   X1: array of features with same units - each column is a feature
        %   X1vars: cell array of X1 feature var names
        % Output formatted similarly to input. Total number of output features is
        %   nchoosek(size(X1,2), 2).
        numdatapoints = size(X1,1);
        binomCoeff = nchoosek(1:length(size(X1,2)), 2);
        numfeaturevars = size(binomCoeff, 1);
        Xout = zeros(numdatapoints,numfeaturevars);
        Xoutvars = cell(1,numfeaturevars);
        idxXout = 1;
        for i = 1:size(binomCoeff,1)
            Xout(:,idxXout) = abs(X1(:, binomCoeff(i,1)) - X1(:, binomCoeff(i,2)));
            Xoutvars{idxXout} = strcat('|', X1vars{binomCoeff(i,1)}, '-', X1vars{binomCoeff(i,2)}, '|');
        end
    end

end