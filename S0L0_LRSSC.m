%
% S0/L0 Low Rank Sparse Subbspace Clustering for estimation of the
% affinity matrix coefficients solves the following optimization problem:
% 
% \min_C 1/2*||X-XC||^2_F+\lambda*||C||_{S_0}+tau*||C||_0  s.t. diag(C)=0.
%
% INPUTS:
%   X: dxN data matrix with d features and N samples
%   lambda: rank regularization constant, sparsity regularization 
%           constant is (1-lambda)
%   mu: inital value of penalty parameter in augmented Lagrangian
%   opts:  Structure for optional parameters with following fields:
%          iter_max: maximum number of iterations
%          err_thr: error threshold for checking convergence condition
%          rho: step size for adaptively changing mu
%          mu_max: maximum value of penalty parameter mu
%
% OUTPUTS:
%   C: NxN matrix of coefficients
%   error: ||X-XC||/||X||
%
% Maria Brbic , January, 2018.
%
function [C, error] = S0L0_LRSSC (X, lambda, mu, opts)

if ~exist('opts', 'var')
    opts = [];
end

% default parameters
rho = 3;
mu_max = 1e6;
err_thr = 1e-4;
iter_max = 100;

tau = 1-lambda;

if isfield(opts, 'iter_max');    iter_max = opts.iter_max;    end
if isfield(opts, 'err_thr');    err_thr = opts.err_thr;    end
if isfield(opts, 'rho');      rho = opts.rho;      end
if isfield(opts, 'mu_max');      mu_max = opts.mu_max;      end

%%

[~,N] = size(X);

C = zeros(N,N);
LAM = zeros(N,N);

XT = X'*X;

Jf = inv(XT+mu*eye(N));
J = Jf*(XT+mu*C-LAM);
J = normc(J); 

not_converged = 1;
iter = 1;

while not_converged
    
    J_prev = J;
    
    % update J
    J = Jf*(XT+mu*C-LAM);
    J = normc(J); 
    
    % update C1
    [U, Sig, V] = svd(J+LAM/mu,'econ');
    sig = diag(Sig)';
    thr = lambda/mu;
    
    thr = sqrt(2*thr);
    sig_thr = sig.*((sign(abs(sig)-thr)+1)/2);
    
    [is, inds] = sort(sig_thr,'descend');
    ind = inds(1:sum(sign(is)));
    sig = sig_thr(ind);
    V = V(:,ind);
    U = U(:,ind);
    Sig = diag(sig);
    C1 = U*Sig*V';
    
    % update C2
    tmp = J+LAM/mu;
    thr = tau/mu;
    thr = sqrt(2*thr);
    C2 = tmp.*((sign(abs(tmp)-thr)+1)/2);
    C2 = C2-diag(diag(C2));
    
    C = lambda*C1+tau*C2;
    
    % update Lagrange multiplier
    LAM = LAM+mu*(J-C);
    
    % update penalty parameter
    mu = min(rho*mu, mu_max);
    
    if rho~=1
        Jf = inv(XT+mu*eye(N));
    end
    
    err1 = max(max(abs(J-C)));
    err2 = max(max(abs(J-J_prev)));
    
    if err1 < err_thr && err2 < err_thr
        not_converged = 0;
    end
    
    % check convergence
    if iter >= iter_max
        not_converged = 0;
    end
    
    iter = iter+1;
    
end

C = normc(C);
error = norm(X-X*J)/norm(X);

