function varargout = monte_carlo(op, varargin)
%% Monte Carlo module: payoff DGP, population regularized target, one replication, one checkpoint.
%   Usage: monte_carlo("<operation>", arguments...). Operations:
%     "simulate"      simulate_payoff_panel
%     "population"    population_payoff
%     "replication"   mc_replication
%     "chunk"         mc_chunk
switch op
    case "simulate", f = @simulate_payoff_panel;
    case "population", f = @population_payoff;
    case "replication", f = @mc_replication;
    case "chunk", f = @mc_chunk;
    otherwise, error("monte_carlo: unknown operation ""%s""", op);
end
[varargout{1:nargout}] = f(varargin{:});
end

function status = mc_chunk(file, fingerprint, T, indices, spec, pop)
%% Run one checkpoint of replications and save it, unless file already holds results for the same specification
%   fingerprint. The file is written atomically (temporary file, then rename), so an interrupted chunk is redone.
if isfile(file)
    S = load(file, "fingerprint");
    if isfield(S, "fingerprint") && S.fingerprint == fingerprint, status = "reused"; return, end
end
results = cell(numel(indices), 1);
for i = 1:numel(indices)
    results{i} = mc_replication(T, indices(i), spec, pop);
end
results = [results{:}];
tmp = file + ".tmp.mat";
save(tmp, "results", "fingerprint", "T", "indices");
movefile(tmp, file);
status = "computed";
end

function r = mc_replication(T, index, spec, pop)
%% One replication of the Monte Carlo study (README, "Monte Carlo study"): one panel of the quadratic payoff DGP,
%   the uncentered fit with eta chosen by chronological CV (as in E1-E4) or, if spec.effdim_target is set, by the
%   effective-dimension rule on the payoff operator, and the regional tests of h' (increasing,
%   decreasing) and h'' (convex, concave). Both derivative orders share the fit and one set of calibration draws.
%   p, W: region x restriction, restrictions ordered [increasing, decreasing, convex, concave].
panel = RandStream("mrg32k3a", "Seed", spec.seed); panel.Substream = 2 * index - 1;
calib = RandStream("mrg32k3a", "Seed", spec.seed); calib.Substream = 2 * index;
t0 = tic;
[X, R, epsilon] = simulate_payoff_panel(T, spec.rho_x, spec.rho_v, spec.sigma_eps, panel);
Y = sum(spec.h0(X) .* R, 2) + epsilon;
E = randn(calib, spec.B, spec.n_points);
select = {};
if isfield(spec, "effdim_target"), select = {"target_effdim", spec.effdim_target}; end
f = estimation("payoff", X, R, Y, spec.eta_grid, spec.n_blocks, spec.chol_tol, spec.x_display, select{:});
k = find(f.eta == f.eta_hat);
g = f; g.beta = f.beta_path(:, k); g.lambda_hat = f.lambda(k); g.eta_hat = f.eta_hat;
t1 = tic;
o1 = inference("regional", g, spec.regions, spec.n_points, spec.B, E);       % h':  [increasing, decreasing]
g.kernel_dx = f.kernel_dxx;
o2 = inference("regional", g, spec.regions, spec.n_points, spec.B, E);       % h'': [convex, concave]
r.time_tests = toc(t1);
r.p = [o1.p_value, o2.p_value]; r.W = [o1.W, o2.W];
r.theta = [o1.theta_hat, o2.theta_hat];                               % h'(xi), h''(xi) at the testing points
r.eta_hat = f.eta_hat; r.lambda_hat = f.lambda(k); r.ell = f.ell; r.rank = f.rank; r.n_lags = o1.n_lags;
r.cv_loss = f.cv_loss; r.cv_loss_intercept = f.cv_loss_intercept;
if isfield(f, "effdim_hat"), r.effdim_hat = f.effdim_hat; end         % attained effective dimension (effdim rule)
r.h = f.h_path(:, k);                                                 % h_hat on the display grid
d = f.kernel(pop.nodes, f.x_piv) * f.c_path(:, k) - spec.h0(pop.nodes);
r.err = sqrt(2 * pop.weights' * d.^2 + 2 * spec.rho_R * (pop.weights' * d)^2);   % ||Sigma^{1/2}(h_hat - h0)||
r.time_chol = f.time_chol; r.time_cv = f.time_cv; r.time_fit = f.time_fit; r.time_total = toc(t0);
end

function [X, R, epsilon] = simulate_payoff_panel(T, rho_x, rho_v, sigma_eps, stream)
%% Two-asset payoff panel (README, "Monte Carlo study"), drawn from the RandStream stream.
%   X (T x 2): characteristics X_j = Phi(U_j), U_j independent stationary Gaussian AR(1) (persistence rho_x, unit variance).
%   R (T x 2): next-period returns R_j = sqrt(3)(2 Phi(V_j) - 1), V iid N(0, [1 rho_v; rho_v 1]): U[-sqrt3, sqrt3] marginals,
%   corr(R_1, R_2) = (6/pi) asin(rho_v/2). epsilon (T x 1): sigma_eps sqrt(3)(2 Phi(W) - 1), W iid N(0, 1).
e = randn(stream, T, 2);
U = zeros(T, 2);
U(1, :) = e(1, :);                                          % stationary start
for t = 2:T
    U(t, :) = rho_x * U(t - 1, :) + sqrt(1 - rho_x^2) * e(t, :);
end
Z = randn(stream, T, 2);
V = [Z(:, 1), rho_v * Z(:, 1) + sqrt(1 - rho_v^2) * Z(:, 2)];
W = randn(stream, T, 1);
X = 0.5 * erfc(-U / sqrt(2));
R = sqrt(3) * erf(V / sqrt(2));                             % 2 Phi(v) - 1 = erf(v / sqrt 2)
epsilon = sigma_eps * sqrt(3) * erf(W / sqrt(2));
end

function pop = population_payoff(h0, ell, rho_R, eta, xi, n_nodes, x_display)
%% Population regularized target h_lambda = (Sigma + lambda I)^{-1} Sigma h0 for the two-asset payoff: characteristics
%   iid U[0,1], unit-variance returns with correlation rho_R independent of X, so Sigma = 2C + 2 rho_R m (x) m with
%   C = E[k(X,.) (x) k(X,.)], m = E[k(X,.)]. Exact for the n_nodes Gauss-Legendre measure on [0,1]: h_lambda =
%   sum_i c_i k(x_i,.) with (A K + lambda I) c = A h0(x), A = 2(W + rho_R w w'), solved as c = M (M K M + lambda)^{-1} M h0
%   with M = A^{1/2}. lambda = eta * tr(Sigma), tr(Sigma) = E||psi||^2 = 2 + 2 rho_R E k(X, X'). Matern-5/2, lengthscale ell.
[x, w] = gauss_legendre01(n_nodes);
[kernel, kernel_dx] = kernels("make", x, ell);
K = kernel(x, x);
A = 2 * (diag(w) + rho_R * (w * w'));
[V, d] = eig((A + A') / 2, "vector");
M = V * diag(sqrt(d)) * V';
MKM = M * K * M;
[U, s] = eig((MKM + MKM') / 2, "vector");
pop.trace = 2 + 2 * rho_R * (w' * K * w);
pop.lambda = eta(:)' * pop.trace;
pop.c = M * (U * ((U' * (M * h0(x))) ./ (s + pop.lambda)));
pop.dh = kernel_dx(xi, x) * pop.c;                          % h_lambda'(xi), one column per eta
pop.d2h = kernels("matern52_dxx", xi, x, ell) * pop.c;                % h_lambda''(xi)
pop.h = kernel(x_display, x) * pop.c;
[pop.x, pop.w, pop.eta, pop.ell, pop.rho_R] = deal(x, w, eta(:)', ell, rho_R);
end

function [x, w] = gauss_legendre01(n)
%% Gauss-Legendre nodes and probability weights on [0, 1] (Golub-Welsch).
b = (1:n - 1) ./ sqrt(4 * (1:n - 1).^2 - 1);
[V, D] = eig(diag(b, 1) + diag(b, -1), "vector");
[t, order] = sort(D);
x = (t + 1) / 2;
w = V(1, order)'.^2;                                        % weights of the probability measure (sum to one)
end
