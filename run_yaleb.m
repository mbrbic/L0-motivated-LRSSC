%
% Run GMC-LRSSC and L0-LRSSC on the Yale B dataset. In each iteration L
% clusters are randomly selected.
%
% INPUTS:
%   L: number of clusters (min: 2, max: 38)
%
% OUTPUTS:
%   CE_stats: mean, std, median and max value of clustering error
%   ET_stats: mean and std of elapsed time
%
% Maria Brbic , January, 2018.
%
function [ CE_stats, ET_stats ] = run_yaleb( L )

addpath datasets/

% for reproducible results, seed the random number generator
s = RandStream('mcg16807','Seed',100);
RandStream.setGlobalStream(s);

%%

load YaleBCrop025.mat   % resized raw images provided along with the SSC codes

Y0 = Y;

n = 64;

n_trials = 100;              % take 100 random subsets of 38 people for each number of clusters

CE  = zeros(2,100);         % clustering error
ET  = zeros(2,100);         % elapsed time

cluster_id = zeros(n_trials, L);

for i_trial = 1:n_trials
    cluster_id(i_trial,:) = randsample(38, L);
end

for i_trial = 1:n_trials
    
    fprintf('Iteration %d\n', i_trial);
    
    %% Choose L random clusters out of 38
    N = n*L;
    
    Y = [];
    for i=1:L
        Y = [Y Y0(:,:,cluster_id(i_trial,i))];
    end
    A0 = reshape(repmat(1:L,n,1),1,N);
    
    %% GMC-LRSSC
    
    fprintf('Running GMC-LRSSC..\n'); i_algo = 1;
    start = tic;
    
    alpha = 1000; mu2 = 3; gamma = 1;
    options = struct('gamma',gamma);
    
    [C, ~] = GMC_LRSSC(normc(Y), alpha, mu2, options);
    A = spectral_clustering(abs(C)+abs(C'),L);
    ET(i_algo,i_trial) = toc(start);
    CE(i_algo,i_trial) = clustering_error(A,A0);
    
    %% S0/L0-LRSSC
    
    fprintf('Running S0/L0-LRSSC..\n'); i_algo = 2;
    start = tic;
    
    lambda = 0.5; mu = 1;
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
        ET_stats = [mean(ET(:,1:i_trial)'); std(ET(:,1:i_trial)')]
    end
    
end


