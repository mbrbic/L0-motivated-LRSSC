%
% GMC Low Rank Sparse Subbspace Clustering for estimation of the
% affinity matrix coefficients solves the following optimization problem:
%
% \min_C 1/2*||X-XC||^2_F+\lambda*psi_B(sigma(C))+tau*psi_B(C)
%     s.t. diag(C)=0.
%
% INPUTS:
%   X: dxN data matrix with d features and N samples
%   alpha: parameter for setting rank and sparsity regularization constants
%   mu2: inital value of penalty parameter for auxiliary variable C2 in the 
%        augmented Lagrangian
%   opts:  Structure for optional parameters with the following fields:
%          gamma: nonconvexity parameter gamma in GMC
%          iter_max: maximum number of iterations
%          err_thr: error threshold for checking convergence condition
%          rho: step size for adaptively changing mu1 and mu2
%          mu1: inital value of penalty parameter for auxiliary variable C1 
%               in augmented Lagrangian
%          mu1_max: maximum value of penalty parameter mu1
%          mu2_max: maximum value of penalty parameter mu2
%
% OUTPUTS:
%   C: NxN matrix of coefficients
%   error: ||X-XC||/||X||
%
% Maria Brbic , January, 2018.
%
function [C, error] = GMC_LRSSC (X, alpha, mu2, opts)

if ~exist('opts', 'var')
    opts = [];
end

% default parameters
lambda = mu2/(1+alpha);
tau = mu2*alpha/(1+alpha);

gamma = 0.6;

mu1 = 0.1;
mu1_max = 1e6;
mu2_max = 1e6;
rho = 3;

iter_max = 100;
err_thr = 1e-4;

if isfield(opts, 'gamma');      gamma = opts.gamma;      end
if isfield(opts, 'iter_max');    iter_max = opts.iter_max;    end
if isfield(opts, 'err_thr');    err_thr = opts.err_thr;    end
if isfield(opts, 'rho');      rho = opts.rho;      end
if isfield(opts, 'mu1');      mu1 = opts.mu1;      end
if isfield(opts, 'mu1_max');      mu1_max = opts.mu_max;      end
if isfield(opts, 'mu2_max');      mu2_max = opts.mu_max;      end

%% initialization

[~,N] = size(X);

C1 = zeros(N,N);
C2 = zeros(N,N);

% Lagrange multpliers
LAM_1 = zeros(N,N);
LAM_2 = LAM_1;

XT = X'*X;

Jf = inv(XT+(mu1+mu2)*eye(N));
J = Jf*(XT+mu1*C1+mu2*C2-LAM_1-LAM_2);
J = normc(J);

not_converged = 1;
iter = 1;

while not_converged
    
    J_prev = J;
    
    % update J
    J = Jf*(XT+mu1*C1+mu2*C2-LAM_1-LAM_2);
    J = normc(J);
    
    % update C1
    [U, Sig, V] = svd(J+LAM_1/mu1,'econ');
    sig = diag(Sig)';
    thr = lambda/mu1;
    a = gamma/thr; % nonconvex rank estimator
    tmp = (sig-lambda/mu1)/(1-a*(lambda/mu1));
    sigm = max([tmp; zeros(1,length(sig))]);
    tmp = [sig; sigm];
    Sig = diag(min(tmp).*sign(sig));
    C1 = U*Sig*V';
    
    % update C2
    tmp = J+LAM_2/mu2;
    thr = tau/mu2;
    a = gamma/thr;
    tmp2 = (abs(tmp)-thr)/(1-a*thr);
    tmp2 = max(tmp2, zeros(size(tmp2)));
    C2 = min(abs(tmp),tmp2).*sign(tmp);
    C2 = C2-diag(diag(C2));
    
    % update Lagrange multipliers
    LAM_1 = LAM_1+mu1*(J-C1);
    LAM_2 = LAM_2+mu2*(J-C2);
    
    % update penalty parameters
    mu1 = min(rho*mu1, mu1_max);
    mu2 = min(rho*mu2, mu2_max);
    
    if rho~=1
        Jf = inv(XT+(mu1+mu2)*eye(N));
    end
    
    err1 = max(max(abs(J-C1)));
    err2 = max(max(abs(J-C2)));
    err3 = max(max(abs(J-J_prev)));
    
    % check convergence
    if iter >= iter_max
        not_converged = 0;
    end
    
    if err1<err_thr && err2<err_thr && err3<err_thr
        not_converged = 0;
    end
    
    iter = iter+1;
    
end

C = normc(C1);
error = norm(X-X*J)/norm(X);

