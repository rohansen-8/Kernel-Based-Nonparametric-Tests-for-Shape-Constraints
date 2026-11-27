function varargout = estimation(op, varargin)
%% Estimation module: RKHS regression in a pivoted-Cholesky basis, chronological cross-validation, and the
%   E1-E4 curve fit ("curve") and Monte Carlo payoff fit ("payoff"). The penalty is chosen over the eta grid either by
%   chronological cross-validation (default) or, with name-value target_effdim = d, by the effective-dimension rule.
%   Usage: estimation("<operation>", arguments...). Operations:
%     "curve"    fit_curve
%     "payoff"   fit_payoff
switch op
    case "curve", f = @fit_curve;
    case "payoff", f = @fit_payoff;
    otherwise, error("estimation: unknown operation ""%s""", op);
end
[varargout{1:nargout}] = f(varargin{:});
end

function f = fit_curve(x_panel, Y_panel, domain, eta_grid, n_blocks, chol_tol, x_grid, A_panel, x_piv, opts)
%% Fit the RKHS regression of a T x J panel on one or two inputs (see make_kernel and fit_krr).
%   x_panel is T x J (one input) or T x J x 2 (maturity, short rate); domain has one row [lo hi] per input.
%   Keeps observations inside the learning domain, uses median-distance lengthscales, selects eta by chronological
%   CV, and evaluates the fit for every candidate eta (h_path) and for eta_hat (h_grid) at the rows of x_grid.
%   Fitted mean a + kernel(x, x_piv) * c, with a = 0 (no intercept). A scalar eta_grid skips cross-validation and
%   fits at that eta.
%   Optional A_panel (T x J, e.g. amount outstanding) gives within-month weights A / (monthly sum of A over kept
%   observations), so every month has total weight one; empty, all observations in a month are weighted equally.
%   Optional x_piv (saved pivot points of an earlier fit on the same data and kernel) rebuilds that reduced basis:
%   L(piv,:) is the Cholesky factor of kernel(x_piv, x_piv) in pivot order, and L = kernel(x, x_piv) / L(piv,:)'.
%   Name-value target_effdim = d replaces cross-validation by select_effdim (eta with effective dimension closest to d).
arguments
    x_panel double
    Y_panel double
    domain double
    eta_grid double
    n_blocks (1, 1) double
    chol_tol (1, 1) double
    x_grid double
    A_panel double = []
    x_piv double = []
    opts.target_effdim double = []
end
effdim = ~isempty(opts.target_effdim);
[T, J, p] = size(x_panel);
period = repmat((1:T)', 1, J);
if isempty(A_panel), A_panel = ones(T, J); end
keep = isfinite(Y_panel) & isfinite(A_panel) & A_panel > 0;
for j = 1:p
    keep = keep & x_panel(:, :, j) >= domain(j, 1) & x_panel(:, :, j) <= domain(j, 2);
end
f.x = zeros(nnz(keep), p);
for j = 1:p
    xj = x_panel(:, :, j);
    f.x(:, j) = xj(keep);
end
f.y = Y_panel(keep);
f.t = period(keep);
A = A_panel(keep);
total = accumarray(f.t, A);
f.w = A ./ total(f.t);                                 % weights sum to one within each month

[f.kernel, f.kernel_dx, f.ell] = kernels("make", f.x);
tic
if isempty(x_piv)
    [f.L, f.piv] = pivoted_cholesky(f.kernel, f.x, chol_tol);
else
    [found, f.piv] = ismember(x_piv, f.x, "rows");
    assert(all(found), "saved pivots are not observations of this fit");
    R = chol(f.kernel(x_piv, x_piv), "lower");
    f.L = f.kernel(f.x, x_piv) / R';
    f.L(f.piv, :) = R;
end
f.time_chol = toc;
if isscalar(eta_grid) || effdim
    [f.eta_hat, f.cv_loss, f.cv_loss_intercept, f.time_cv] = deal(eta_grid, [], [], 0);
else
    tic; [f.eta_hat, f.cv_loss, f.cv_loss_intercept] = cv_lambda(f.x, f.y, f.w, f.t, eta_grid, n_blocks, chol_tol); f.time_cv = toc;
end
tic; [a_path, beta_path, lambda_path, mu] = fit_krr(f.L, f.y, f.w, eta_grid); f.time_fit = toc;   % one eigendecomposition
if effdim                                              % reuses the fit's eigenvalues: no further decomposition
    [f.eta_hat, f.effdim_path] = select_effdim(mu, lambda_path, eta_grid, opts.target_effdim);
end
m = find(eta_grid == f.eta_hat);
f.x_piv = f.x(f.piv, :);
c_path = f.L(f.piv, :)' \ beta_path;
f.x_grid = x_grid;
f.h_path = a_path + f.kernel(x_grid, f.x_piv) * c_path;
[f.a, f.beta, f.lambda_hat, f.c, f.h_grid] = deal(a_path(m), beta_path(:, m), lambda_path(m), c_path(:, m), f.h_path(:, m));
f.T = numel(unique(f.t));
if effdim, [f.effdim_target, f.effdim_hat] = deal(opts.target_effdim, f.effdim_path(m)); end
end

function f = fit_payoff(X, R, Y, eta_grid, n_blocks, chol_tol, x_grid, opts)
%% Uncentered RKHS regression for the portfolio payoff R(h; Z_t) = sum_j h(X_tj) R_tj, one observation per period.
%   X, R: T x J characteristics and returns. Y: T x m targets sharing every factorization (e.g. several configurations).
%   Basis: pivoted Cholesky on the pooled T*J characteristic points; period features F = S L with S(t, (t,j)) = R_tj, so
%   F(t,:) represents psi_t = sum_j R_tj k(X_tj, .). Equal period weights; lambda = eta * tr(Sigma_hat) (fit_krr), with
%   tr(Sigma_hat) = mean_t ||psi_t||^2. opts.ell fixes the lengthscale (default: median distance of the pooled points);
%   opts.lambda fixes the penalties (no CV). opts.target_effdim = d chooses one eta for all target columns by select_effdim
%   on this payoff operator (actual lambda = eta * tr(Sigma), tr(Sigma) not one). Otherwise eta is chosen per target
%   column by chronological CV over eta_grid.
%   Returns the path (beta, c, h at x_grid: r x k x m) and the fields regional_tests uses (L, L_piv, x_piv, kernel_dx, ...).
arguments
    X double
    R double
    Y double
    eta_grid double
    n_blocks (1, 1) double
    chol_tol (1, 1) double
    x_grid double
    opts.ell double = []
    opts.lambda double = []
    opts.target_effdim double = []
end
[T, J] = size(X);
x = reshape(X', [], 1);                                     % points ordered by period, then asset
S = sparse(repelem((1:T)', J), (1:T * J)', reshape(R', [], 1), T, T * J);
f = struct("T", T, "t", (1:T)', "w", ones(T, 1), "y", Y, "a", 0);
[f.kernel, f.kernel_dx, f.ell, f.kernel_dxx] = kernels("make", x, opts.ell);   % kernel_dxx: second derivative (convexity tests)
tic; [L, f.piv] = pivoted_cholesky(f.kernel, x, chol_tol); f.time_chol = toc;
f.L = S * L; f.L_piv = L(f.piv, :); f.x_piv = x(f.piv); f.rank = numel(f.piv);
f.trace_Sigma = mean(sum(f.L.^2, 2));
if ~isempty(opts.lambda)
    f.eta = opts.lambda(:)' / f.trace_Sigma; f.eta_hat = []; f.time_cv = 0;
elseif ~isempty(opts.target_effdim)
    f.eta = eta_grid(:)'; [f.cv_loss, f.cv_loss_intercept, f.time_cv] = deal([], [], 0);
else
    tic; [f.eta_hat, f.cv_loss, f.cv_loss_intercept] = cv_lambda(x, Y, f.w, f.t, eta_grid, n_blocks, chol_tol, S);
    f.time_cv = toc; f.eta = eta_grid(:)';
end
tic; [~, f.beta_path, f.lambda, mu] = fit_krr(f.L, Y, f.w, f.eta); f.time_fit = toc;   % one eigendecomposition
if isempty(opts.lambda) && ~isempty(opts.target_effdim)      % design-only rule: one eta for every target column
    [eta_hat, f.effdim_path] = select_effdim(mu, f.lambda, f.eta, opts.target_effdim);
    f.eta_hat = repmat(eta_hat, 1, size(Y, 2));
    [f.effdim_target, f.effdim_hat] = deal(opts.target_effdim, f.effdim_path(f.eta == eta_hat));
end
K_grid = f.kernel(x_grid, f.x_piv);
f.c_path = zeros(size(f.beta_path)); f.h_path = zeros(numel(x_grid), numel(f.eta), size(Y, 2));
for c = 1:size(Y, 2)
    f.c_path(:, :, c) = f.L_piv' \ f.beta_path(:, :, c);
    f.h_path(:, :, c) = K_grid * f.c_path(:, :, c);
end
end

function [a, beta, lambda, mu] = fit_krr(L, y, w, eta)
%% Weighted RKHS regression in the reduced basis of pivoted_cholesky, for every value in eta.
%   h(x) = l(x)'beta with ||h||_H = ||beta||; weights are normalized to sum to one. E[Y|x] = h(x), minimizing
%       (1/2) sum_i w_i (y_i - l_i'beta)^2 + (lambda/2) ||beta||^2,   Sigma = sum_i w_i l_i l_i'.
%   Penalty lambda = eta * tr(Sigma), the trace of the weighted second-moment operator of the features in the loss;
%   tr(Sigma) = sum_i w_i k(x_i, x_i) = 1 up to the basis tolerance (unit kernel diagonal), so lambda = eta.
%   y may hold several response columns sharing the features; then beta is r x k x m. a = 0 (no intercept), 1 x k x m.
%   mu: eigenvalues of Sigma (the nonzero eigenvalues of the weighted Gram operator), for select_effdim.
w = w(:) / sum(w);
m = size(y, 2);
Sigma = L' * (w .* L);
lambda = eta(:)' * trace(Sigma);
g = L' * (w .* y);
[V, D] = eig((Sigma + Sigma') / 2, "vector");
beta = zeros(size(L, 2), numel(eta), m); a = zeros(1, numel(eta), m);
for c = 1:m
    beta(:, :, c) = V * ((V' * g(:, c)) ./ (D + lambda));
end
mu = max(D, 0);
end

function [eta_hat, N] = select_effdim(mu, lambda, eta, d)
%% Effective-dimension rule: eta_hat = argmin over the grid of |N(lambda) - d|, exact ties to the larger penalty.
%   N(lambda) = sum_j mu_j / (mu_j + lambda), with mu the eigenvalues of the weighted covariance Sigma entering the fit
%   and lambda = eta * tr(Sigma) the actual ridge penalties of the grid (fit_krr). Uses only the design, not y.
N = sum(mu(:) ./ (mu(:) + lambda(:)'), 1);
gap = abs(N - d);
tied = find(gap == min(gap));
[~, i] = max(eta(tied));
eta_hat = eta(tied(i));
end

function [eta_hat, loss, loss_intercept] = cv_lambda(x, y, w, period, eta, n_blocks, chol_tol, S)
%% Chronological expanding-window cross-validation for the dimensionless penalty eta (inputs x: n x 1 or n x 2).
%   Periods are split into n_blocks consecutive blocks; fold k trains on blocks 1..k and validates on block k+1.
%   Each fold uses its own training lengthscales (make_kernel), reduced basis and trace scaling.
%   loss(k, m, c): weighted validation MSE of fold k at eta(m) for response column c of y; loss_intercept(k, c):
%   same for the training mean alone (reference only).
%   Optional S (observations x points, sparse) is a payoff aggregation: observation i has features sum_q S(i,q) l(x_q),
%   and the rows of x are the evaluation points. Without S each observation is one point (point evaluation).
payoff = nargin >= 8 && ~isempty(S);
periods = unique(period);
edges = round(linspace(0, numel(periods), n_blocks + 1));
m = size(y, 2);
loss = zeros(n_blocks - 1, numel(eta), m);
loss_intercept = zeros(n_blocks - 1, m);
for k = 1:n_blocks - 1
    train = period <= periods(edges(k + 1));
    valid = period > periods(edges(k + 1)) & period <= periods(edges(k + 2));
    if payoff
        tp = full(any(S(train, :), 1))'; vp = full(any(S(valid, :), 1))';   % points of the training/validation periods
    else
        tp = train; vp = valid;
    end
    x_train = x(tp, :);
    kernel = kernels("make", x_train);
    [L, piv] = pivoted_cholesky(kernel, x_train, chol_tol);
    L_valid = kernel(x(vp, :), x_train(piv, :)) / L(piv, :)';
    if payoff, L = S(train, tp) * L; L_valid = S(valid, vp) * L_valid; end
    [a, beta] = fit_krr(L, y(train, :), w(train), eta);
    for c = 1:m
        resid = y(valid, c) - (a(:, :, c) + L_valid * beta(:, :, c));
        loss(k, :, c) = (w(valid)' * resid.^2) / sum(w(valid));
        ybar = (w(train)' * y(train, c)) / sum(w(train));
        loss_intercept(k, c) = (w(valid)' * (y(valid, c) - ybar).^2) / sum(w(valid));
    end
end
[~, idx] = min(mean(loss, 1), [], 2);
eta_hat = reshape(eta(idx), 1, []);
end

function [L, piv] = pivoted_cholesky(kernel, x, tol)
%% Pivoted Cholesky factor K ~ L*L' of the Gram matrix K = kernel(x, x), built column by column.
%   kernel(x, xi) returns the kernel column for the rows xi of x; kernel diagonals equal one.
%   Stops when the largest residual diagonal falls below tol, so 0 <= K - L*L' and max diag(K - L*L') <= tol.
n = size(x, 1);
d = ones(n, 1);
L = zeros(n, min(n, 500));
piv = zeros(n, 1);
r = 0;
while max(d) > tol && r < n
    [dmax, i] = max(d);
    r = r + 1;
    if r > size(L, 2), L(:, 2 * size(L, 2)) = 0; end
    col = kernel(x, x(i, :)) - L(:, 1:r - 1) * L(i, 1:r - 1)';
    L(:, r) = col / sqrt(dmax);
    piv(r) = i;
    d = max(d - L(:, r).^2, 0);
end
L = L(:, 1:r);
piv = piv(1:r);
end
