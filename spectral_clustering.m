function A = spectral_clustering(Z,L)
N = size(Z,1);

DN = diag(1./sqrt(sum(Z)+eps));
LapN = speye(N) - DN*Z*DN;
[~,~,vN] = svd(LapN);
kerN = vN(:,N-L+1:N);
for i = 1:N
    KerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
end

A = kmeans(KerNS,L,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
