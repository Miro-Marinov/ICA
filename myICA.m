% load digit
% for i = 1 : length(train)
%     trainData(:,i) = reshape(train{i},784,1);
% end
% data = trainData;
% 
% for i = 1 : length(test)
%     testData(:,i) = reshape(test{i},1,784);
% end

function [S,w,whiteningMat,meanData] = myICA( data, components)

% Preprocess the data

% Convert data to double
data = double(data);
    
[features, examples] = size(data);


% Step 1 Center the data
meanData = mean(data,2);
data = data - repmat(meanData,1,examples);


% Step 2 Whiten the data
% This removes all linear dependencies
covMat = cov(data');
[eigVec,eigVal,~] = svd(covMat);
% Next line is from
% http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening
% Whitening transformation should be eigVal^-1/2 * eigVec^T
% whiteningMat = eigVec * diag(1 ./ sqrt(diag(eigVal))) * eigVec';
% dataW = whiteningMat * data;
dataW = (sqrtm(eigVal) \ eigVec') * data;

% Step 3 Chose intiial random vector w

% Make it a unit vector
% Step 4 calculate new w from w0
% FastICA algorithm

for p = 1 : components
    w(:,p) = randn(features,1);
%     w(p) = w(p)/norm(w(p),2);
    while(1)
        wPrev = w(:,p);
        g = tanh(w(:,p)' * dataW)';
        gD = 1 - tanh(w(:,p)' * dataW).^2;
        w(:,p) = 1 / examples * (dataW * g) - ( 1 / examples * gD) * ones(examples,1) * w(:,p);
        sum = 0;
        for j = 1 : p - 1
            sum = sum + w(:,j)*w(:,p)'*w(:,j);
        end
        w(:,p) = w(:,p) - sum;
        w(:,p) = w(:,p)/norm(w(:,p),2);
        if abs(abs(wPrev'*w(:,p))-1) <= 0.000001
            break;
        end
    end
end
w = w';
S = w * dataW;

end
    
    

    





