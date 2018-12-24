%
% Run GMC-LRSSC and L0-LRSSC on the USPS dataset. In each iteration num_im
% images per digit are randomly selected.
%
% INPUTS:
%   digits: set of digits (e.g. [3,6,9])
%
% OUTPUTS:
%   CE_stats: mean, std, median and max value of clustering error
%   ET_stats: mean and std of elapsed time
%
% Maria Brbic , January, 2018.
%
function [ CE_stats, ET_stats ] = run_usps( digits )

addpath datasets/

% for reproducible results, seed the random number generator
s = RandStream('mcg16807','Seed',100);
RandStream.setGlobalStream(s);

%% Problem parameters

num_im = 50; % number of images in each subspace

n_trial = 100; % number of iterations

CE  = zeros(2,100);         % clustering error
ET  = zeros(2,100);         % elapsed time

digits = digits+1; % clusters start from 1

L = length(digits); % number of clusters


%% test images

load usps
images = data(:,2:end)';
labels = data(:,1)-1;

[labelssorted,IX] = sort(labels);
imgssorted = images(:,IX);

% beg, endd contain the indices of the begin (end) indices of the numbers
beg(1) = 1;
k = 1;
beg(k) = 1;
for i =1:size(images,2) % for each point
    if labelssorted(i) == k-1
    else
        endd(k) = i-1;
        k = k+1;
        beg(k) = i;
    end
end
endd(k) = size(images,2);

for i_trial = 1:n_trial
    
    fprintf('Iteration %d\n', i_trial);
    
    % generate a problem instance
    A0 = [];
    Y = [];
    for l=1:L
        U = beg(digits(l)):endd(digits(l));
        if(n_trial == 1)
            V = U(1:num_im); % take the first n images of each number
        else
            pn = randperm(length(U));
            V = U(pn(1:num_im)); % n random indices drawn without replacement from
        end
        
        A0 = [A0, ones(1,num_im)*l];
        Y = [Y, imgssorted(:,V)];
    end
    
    A0 = A0'; % ground truth
    
    %% GMC-LRSSC
    
    fprintf('Running GMC-LRSSC..\n'); i_algo = 1;
    start = tic;
    
    alpha = 0.1; mu2 = 1; gamma = 0.1;
    options = struct('gamma',gamma);
    
    [C, ~] = GMC_LRSSC(normc(Y), alpha, mu2, options);
    A = spectral_clustering(abs(C)+abs(C'),L);
    ET(i_algo,i_trial) = toc(start);
    CE(i_algo,i_trial) = clustering_error(A,A0);
    
    %% S0/L0-LRSSC
    
    fprintf('Running S0/L0-LRSSC..\n'); i_algo = 2;
    start = tic;
    
    lambda = 0.3; mu = 5;
    [C, ~] = S0L0_LRSSC(normc(Y), lambda, mu);
    A = spectral_clustering(abs(C)+abs(C'), L);
    ET(i_algo,i_trial) = toc(start);
    CE(i_algo,i_trial) = clustering_error(A,A0);
    
    %%
    
    if i_trial == 1
        CE_stats = CE(:,1)'
        ET_stats = ET(:,1)'
    else
        CE_stats = [mean(CE(:,1:i_trial)'); std(CE(:,1:i_trial)'); median(CE(:,1:i_trial)'); max(CE(:,1:i_trial)')]
        ET_stats = [mean(ET(:,1:i_trial)'); std(ET(:,1:i_trial)'); ]
    end
    
end

