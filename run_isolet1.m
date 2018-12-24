%
% Run GMC-LRSSC and L0-LRSSC on the Isolet1 dataset. In each iteration L
% clusters are randomly selected.
%
% INPUTS:
%   L: number of clusters (min: 2, max: 26)
%
% OUTPUTS:
%   CE_stats: mean, std, median and max value of clustering error
%   ET_stats: mean and std of elapsed time
%
% Maria Brbic , January, 2018.
%
function [ CE_stats, ET_stats ] = run_isolet1( L )

addpath datasets/

s = RandStream('mcg16807','Seed',100);
RandStream.setGlobalStream(s);

%% Load data

data = load('datasets/Isolet1.mat');

Y0 = data.fea';

n_trial = 100;    % Take 100 random subsets of 26 people for each number of clusters

CE  = zeros(2,100);         % clustering error
ET  = zeros(2,100);         % elapsed time

n = 60;  % number of images per class

L_max = 26;

cluster_id = zeros(n_trial,L);

for i_trial = 1:n_trial
    cluster_id(i_trial,:) = randsample(L_max,L);
end

for i_trial = 1:n_trial
    
    fprintf('Iteration %d\n', i_trial);
    
    %% Choose L random clusters out of 40
    N = n*L;
    
    Y = [];
    for i=1:L
        id = cluster_id(i_trial,i);
        Y = [Y Y0(:,((id-1)*n+1):((id-1)*n+n))];
    end
    A0 = reshape(repmat(1:L,n,1),1,N);
    
    %% GMC-LRSSC
    
    fprintf('Running GMC-LRSSC..\n'); i_algo = 1;
    start = tic;
    
    alpha = 0.01; mu2 = 20; gamma = 0.1;
    options = struct('gamma',gamma);
    
    [C, ~] = GMC_LRSSC(normc(Y), alpha, mu2, options);
    A = spectral_clustering(abs(C)+abs(C'),L);
    ET(i_algo,i_trial) = toc(start);
    CE(i_algo,i_trial) = clustering_error(A,A0);
    
    %% S0/L0-LRSSC
    
    fprintf('Running S0/L0-LRSSC..\n'); i_algo = 2;
    start = tic;
    
    lambda = 0.7; mu = 20;
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
        ET_stats =[mean(ET(:,1:i_trial)'); std(ET(:,1:i_trial)')]
    end
    
end


