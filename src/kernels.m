function varargout = kernels(op, varargin)
%% Kernel module: Matern-5/2 kernel and its derivatives, Laplacian kernel, kernel construction with
%   median-pairwise-distance lengthscales.
%   Usage: kernels("<operation>", arguments...). Operations:
%     "make"              make_kernel
%     "matern52"          matern52
%     "matern52_dx"       matern52_dx
%     "matern52_dxx"      matern52_dxx
%     "laplace"           laplace_kernel
%     "median_distance"   median_pairwise_distance
switch op
    case "make", f = @make_kernel;
    case "matern52", f = @matern52;
    case "matern52_dx", f = @matern52_dx;
    case "matern52_dxx", f = @matern52_dxx;
    case "laplace", f = @laplace_kernel;
    case "median_distance", f = @median_pairwise_distance;
    otherwise, error("kernels: unknown operation ""%s""", op);
end
[varargout{1:nargout}] = f(varargin{:});
end

function [kernel, kernel_dx, ell, kernel_dxx] = make_kernel(X, ell)
%% Kernel with median-pairwise-distance lengthscales for inputs X (n x 1 or n x 2), and its derivative in X(:,1);
%   for one input also its second derivative (kernel_dxx, empty for two inputs).
%   One input: Matern-5/2. Two inputs (maturity, short rate): Matern-5/2 in maturity times Laplacian in the rate.
%   Optional ell fixes the lengthscales instead (e.g. a population value).
if nargin < 2 || isempty(ell)
    ell = zeros(1, size(X, 2));
    for j = 1:size(X, 2)
        ell(j) = median_pairwise_distance(X(:, j));
    end
end
if size(X, 2) == 1
    kernel = @(a, b) matern52(a, b, ell);
    kernel_dx = @(a, b) matern52_dx(a, b, ell);
    kernel_dxx = @(a, b) matern52_dxx(a, b, ell);
else
    kernel = @(a, b) matern52(a(:, 1), b(:, 1), ell(1)) .* laplace_kernel(a(:, 2), b(:, 2), ell(2));
    kernel_dx = @(a, b) matern52_dx(a(:, 1), b(:, 1), ell(1)) .* laplace_kernel(a(:, 2), b(:, 2), ell(2));
    kernel_dxx = [];
end
end

function K = matern52(x1, x2, ell)
%% Matern-5/2 kernel matrix K(i,j) = k(x1(i), x2(j)) for scalar inputs and lengthscale ell.
r = sqrt(5) * abs(x1(:) - x2(:).') / ell;
K = (1 + r + r.^2 / 3) .* exp(-r);
end

function Kx = matern52_dx(x1, x2, ell)
%% Derivative of the Matern-5/2 kernel in its first argument: Kx(i,j) = d k(x1(i), x2(j)) / d x1(i).
u = x1(:) - x2(:).';
s = sqrt(5) * abs(u) / ell;
Kx = -(5 / (3 * ell^2)) * u .* (1 + s) .* exp(-s);
end

function K = matern52_dxx(x1, x2, ell)
%% Second derivative in x1 of the Matern-5/2 kernel: K(i,j) = d^2/dx1^2 k(x1(i), x2(j)) for scalar inputs.
%   With s = sqrt(5)|x1 - x2|/ell: (5 / (3 ell^2)) (s^2 - s - 1) exp(-s). The kernel is C^4, so this is the
%   representer of second-derivative evaluation (bounded for every x1, including x1 = x2).
s = sqrt(5) * abs(x1(:) - x2(:).') / ell;
K = 5 / (3 * ell^2) * (s.^2 - s - 1) .* exp(-s);
end

function K = laplace_kernel(x1, x2, ell)
%% Laplacian (Matern-1/2) kernel matrix K(i,j) = exp(-|x1(i) - x2(j)| / ell) for scalar inputs.
K = exp(-abs(x1(:) - x2(:).') / ell);
end

function ell = median_pairwise_distance(x)
%% Median of |x_i - x_j| over all pairs i < j (kernel lengthscale rule).
%   Up to 2,000 points the pairwise distances are formed directly. For larger inputs the same median of the same
%   floating-point differences is found without storing them: the two middle order statistics are located by binary
%   search over doubles, counting pairs with fl(x_j - x_i) <= d in O(n log n) memory-light passes.
x = x(:);
n = numel(x);
if n <= 2000
    D = abs(x - x.');
    ell = median(D(triu(true(n), 1)));
    return
end
xs = sort(x);
N = n * (n - 1) / 2;
k = floor((N + 1) / 2);
a = kth_difference(xs, k);
if mod(N, 2) == 1
    ell = a;
else
    b = kth_difference(xs, k + 1);
    ell = a + (b - a) / 2;                                  % as MATLAB's median for an even count
end
end

function v = kth_difference(xs, k)
%% k-th smallest of fl(x_j - x_i), i < j, for sorted xs: the smallest double d with count(d) >= k.
lo = uint64(0);                                             % bit patterns of non-negative doubles are ordered
hi = typecast(xs(end) - xs(1), "uint64");
while lo < hi
    mid = lo + idivide(hi - lo, uint64(2));                % floor((lo + hi) / 2) without overflow
    if pairs_at_most(xs, typecast(mid, "double")) >= k, hi = mid; else, lo = mid + 1; end
end
v = typecast(lo, "double");
end

function c = pairs_at_most(xs, d)
%% Number of pairs i < j with fl(xs(j) - xs(i)) <= d. For each i the count is a prefix of j (differences are
%   monotone in j); a stable merge gives the prefix length for xs(j) <= xs(i) + d, then exact boundary corrections.
n = numel(xs);
[~, order] = sort([xs; xs + d]);                            % stable: xs(j) precedes an equal xs(i) + d
is_x = order <= n;
pos = cumsum(is_x);
ub = zeros(n, 1);
ub(order(~is_x) - n) = pos(~is_x);                          % #{j : xs(j) <= fl(xs(i) + d)}
i = (1:n)';
ub = max(ub, i);
while true                                                  % shrink where the last included difference exceeds d
    bad = ub > i & xs(ub) - xs(i) > d;
    if ~any(bad), break, end
    ub(bad) = ub(bad) - 1;
end
while true                                                  % extend where the next difference is still <= d
    nxt = min(ub + 1, n);
    good = ub < n & xs(nxt) - xs(i) <= d;
    if ~any(good), break, end
    ub(good) = ub(good) + 1;
end
c = sum(ub - i);
end
