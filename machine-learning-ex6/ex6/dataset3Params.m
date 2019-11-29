function [C, sigma] = dataset3Params(X, y, Xval, yval)
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. 

% The values of C and sigma to be tested
C_trial = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_trial = [0.01 0.03 0.1 0.3 1 3 10 30];
% Number of trials for C and sigma
m = size(C_trial,2);


missShot = 1;

for i=1:m
    for j=1:m
        model = svmTrain(X, y, C_trial(i), @(x1, x2) gaussianKernel(x1, x2, sigma_trial(j)));
        
        pred = svmPredict(model, Xval);
        
        if (missShot > mean(double(pred ~= yval))) 
            missShot = mean(double(pred ~= yval));
            C_temp = C_trial(i);
            sigma_temp = sigma_trial(j);
        end    
    end    
end

C = C_temp;
sigma = sigma_temp;

end
