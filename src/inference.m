function varargout = inference(op, varargin)
%% Inference module: monthly influence scores, Newey-West covariance, the cone-Wald statistic with
%   vertex calibration (exact orthant projection), and the regional shape tests.
%   Usage: inference("<operation>", arguments...). Operations:
%     "scores"       krr_scores
%     "newey_west"   newey_west
%     "shape_test"   shape_test
%     "regional"     regional_tests
switch op
    case "scores", f = @krr_scores;
    case "newey_west", f = @newey_west;
    case "shape_test", f = @shape_test;
    case "regional", f = @regional_tests;
    otherwise, error("inference: unknown operation ""%s""", op);
end
[varargout{1:nargout}] = f(varargin{:});
end

function out = regional_tests(f, regions, n_points, B, seed, states)
%% Regional cone-Wald tests of H0: h weakly increasing and H0: h weakly decreasing, for a fit from fit_curve.
%   Each region (row of regions, in regression-input units) gets n_points equally spaced derivative points,
%   endpoints included. One set of monthly scores and one Newey-West covariance serve all regions and both
%   directions; the same B Gaussian draws calibrate every region (vertex law), as in shape_test.
%   Optional states (two-input fits): fixed values of the second input. Derivatives are taken in the first input
%   along each state, and every region is tested at every state; W and p_value are region x direction x state.
%   seed is either a seed for the B x n_points calibration draws or the draws themselves. For a payoff fit
%   (fit_payoff), f.L holds the period features and f.L_piv the point-basis factor used for derivatives.
if nargin < 6, states = []; end
n_regions = size(regions, 1);
xi = zeros(n_regions * n_points, 1);
for r = 1:n_regions
    xi((r - 1) * n_points + (1:n_points)) = linspace(regions(r, 1), regions(r, 2), n_points);
end
n_states = max(numel(states), 1);
if ~isempty(states), xi = [repmat(xi, n_states, 1), repelem(states(:), numel(xi))]; end
if isfield(f, "L_piv"), L_piv = f.L_piv; else, L_piv = f.L(f.piv, :); end
D = f.kernel_dx(xi, f.x_piv) / L_piv';                      % h'(xi) = D * beta (derivative in the first input)
theta_hat = D * f.beta;
G = krr_scores(f.L, f.y, f.w, f.t, f.a, f.beta, f.lambda_hat, D);
n_lags = floor(4 * (f.T / 100)^(2/9));
Omega = newey_west(G, n_lags);
if isscalar(seed)
    rng(seed, "twister"); E = randn(B, n_points);
else
    E = seed; seed = NaN;                                   % precomputed draws, shared by the caller
end
W = zeros(n_regions, 2, n_states); p_value = W; cond_Omega = zeros(n_regions, n_states); W_null = cell(n_regions, n_states);
for s = 1:n_states
    for r = 1:n_regions
        idx = ((s - 1) * n_regions + r - 1) * n_points + (1:n_points);
        [W(r, :, s), p_value(r, :, s), W_null{r, s}] = shape_test([theta_hat(idx), -theta_hat(idx)], Omega(idx, idx), f.T, E);
        cond_Omega(r, s) = cond(Omega(idx, idx));
    end
end
out = struct("regions", regions, "states", states, "xi", xi, "n_points", n_points, "theta_hat", theta_hat, "D", D, ...
    "G", G, "Omega", Omega, "n_lags", n_lags, "B", B, "seed", seed, "W", W, "p_value", p_value, "W_null", {W_null}, ...
    "cond_Omega", cond_Omega, "eta", f.eta_hat, "lambda", f.lambda_hat, "T", f.T);
end

function G = krr_scores(L, y, w, period, a, beta, lambda, D)
%% Period-level grid scores of the RKHS regression in the reduced basis (see fit_krr).
%   G(t,:)' = D (Sigma + lambda I)^{-1} s_t, with s_t = sum_{i in t} q_i l_i e_i - lambda*beta, q_i = T p_i the
%   within-period weights, e_i = y_i - a - l_i'beta (a = 0) and Sigma = sum_i p_i l_i l_i'.
%   s_t is the influence of period t on beta at fixed lambda; the s_t average to zero by the normal equations.
%   This is the analogue of the manuscript's F_t = (Y_t - <h,psi_t>) psi_t - lambda h (4.22), (F.13).
p = w(:) / sum(w);
[~, ~, t] = unique(period);
T = max(t);
e = y - a - L * beta;
Sigma_lambda = L' * (p .* L) + lambda * eye(size(L, 2));
S = zeros(T, size(L, 2));
for k = 1:size(L, 2)
    S(:, k) = accumarray(t, T * p .* L(:, k) .* e, [T 1]);
end
S = S - lambda * beta';
G = S * (Sigma_lambda \ D');
end

function Omega = newey_west(G, n_lags)
%% Bartlett (Newey-West) long-run covariance of the rows of G (T x n), as in manuscript (4.24)-(4.25).
T = size(G, 1);
Omega = G' * G / T;
for l = 1:n_lags
    Gamma = G(1:T - l, :)' * G(1 + l:T, :) / T;
    Omega = Omega + (1 - l / (n_lags + 1)) * (Gamma + Gamma');
end
end

function [W, p_value, W_null] = shape_test(theta_hat, Omega, T, E)
%% One-sided cone Wald test of H0: theta >= 0 (manuscript (4.27), (F.18)-(F.19)), calibrated at the vertex.
%   theta_hat: n x k signed derivative vectors sharing the long-run covariance Omega (e.g. [h', -h']).
%   W(m) = T min_{w >= 0} (theta - w)' Omega^{-1} (theta - w), by exact face enumeration (cone_distance).
%   Vertex law: W_null(b) = min_{w >= 0} (z_b - w)' Omega^{-1} (z_b - w) with z_b = chol(Omega)' E(b,:)' ~ N(0, Omega)
%   for E standard normal (B x n). p_value = (1 + #{W_null >= W}) / (B + 1).
W = zeros(1, size(theta_hat, 2));
for m = 1:size(theta_hat, 2)
    if all(theta_hat(:, m) >= 0), continue, end         % already in the cone: W = 0 exactly
    W(m) = T * cone_distance(theta_hat(:, m), Omega);
end
W_null = cone_distance(chol(Omega)' * E', Omega)';
p_value = (1 + sum(W_null >= W, 1)) / (size(E, 1) + 1);
end

function d = cone_distance(Z, Omega)
%% Squared Mahalanobis distance from each column z of Z (n x m) to the nonnegative orthant:
%   d = min_{w >= 0} (z - w)' Omega^{-1} (z - w), computed exactly by enumerating the 2^n faces of the orthant.
%   On face F (free coordinates; w = 0 elsewhere) the stationary point is w_F = z_F + P_FF^{-1} P_FN z_N with
%   P = Omega^{-1}; d is the smallest distance among faces whose stationary point satisfies w_F >= 0.
%   Works on the correlation scale (the orthant is invariant to positive rescaling of coordinates), so the
%   feasibility tolerance is unit-free. d = 0 exactly only for points in the orthant.
[n, m] = size(Z);
assert(n <= 15, "face enumeration is intended for low-dimensional regional tests");
s = sqrt(diag(Omega));
Z = Z ./ s;
P = inv(Omega ./ (s * s'));
P = (P + P') / 2;
d = inf(1, m);
d(all(Z >= 0, 1)) = 0;                                 % all coordinates free: the point itself
for mask = 0:2^n - 2
    free = bitget(mask, 1:n) == 1;
    R = Z;                                             % residual z - w: z on fixed coordinates, z_F - w_F on free
    R(free, :) = -P(free, free) \ (P(free, ~free) * Z(~free, :));
    feasible = all(Z(free, :) - R(free, :) >= -1e-12, 1);
    dist = sum(R .* (P * R), 1);
    d(feasible) = min(d(feasible), dist(feasible));
end
end
