%% Table F.1: sensitivity of the 1973-2001 Treasury tests (5-8 months) to the effective-dimension target d in {3, 5, 7}.
%   Requires the E3 and E4 results of run_empirical.m and the Treasury panel (data/processed/treasury_panel.mat).
%   The rows for d = 5 are the paper's main results (e3_tests, e4_tests). Each other row refits at the grid penalty that
%   the target selects, reusing the saved pivots of the d = 5 fit (the pivoted Cholesky basis does not depend on eta),
%   and runs the regional tests with the settings of run_empirical.m (regions, points, B, seed, Newey-West, states).
%   The conditional specification selects the same penalty for d = 5 and d = 7, so only three refits are needed.
%   Refits are saved in results/sensitivity/ and reused; the table is printed from the saved results.
root = fileparts(fileparts(mfilename("fullpath")));          % repository root (this file is in scripts/)
addpath(fullfile(root, "src"));
base = fullfile(root, "results");
out_dir = fullfile(root, "results", "sensitivity");
if ~isfolder(out_dir), mkdir(out_dir); end
n_blocks = 4; chol_tol = 1e-10; B = 10000; n_points = 5;   % run_empirical.m settings
regions = [1 5; 5 8; 8 12]; states = [3.64 5.73 9.56];    % maturity regions (months); 1973-2001 short-rate states, % p.a.
runs = struct("name", {"e3", "e3", "e4"}, "d", {3, 7, 3}, "eta", {10^-1, 10^-2.5, 10^-0.5});
run_file = @(r) fullfile(out_dir, sprintf("%s_1973_2001_d%d_eta%+.1f.mat", r.name, r.d, log10(r.eta)));

load(fullfile(base, "e3_fit.mat"), "e3"); load(fullfile(base, "e3_tests.mat"), "e3_tests");
load(fullfile(base, "e4_1973_2001_fit.mat"), "e4"); load(fullfile(base, "e4_1973_2001_tests.mat"), "e4_tests");
s3 = e3(2); t3 = e3_tests(2);                                % 1973-2001 subsample
assert(isequal(s3.subsample, [1973 2001]) && isequal(t3.subsample, [1973 2001]), "e3(2) is not the 1973-2001 subsample");

%% Refits for the targets whose penalty differs from d = 5 (existing files are reused)
todo = runs(arrayfun(@(r) ~isfile(run_file(r)), runs));
if ~isempty(todo)
    load(fullfile(root, "data", "processed", "treasury_panel.mat"), "tp");
    d = tp(year(tp.formation_date) >= 1973 & year(tp.formation_date) <= 2001, :);
    [X, Yp, A, ~, R] = panels("treasury_month", d);
end
for r = todo
    tic
    if r.name == "e3"
        s = s3;
        f = estimation("curve", X, Yp, s.domain, r.eta, n_blocks, chol_tol, s.x_grid, A, s.x_piv);
        res = inference("regional", f, regions, n_points, B, 1);
    else
        s = e4;
        f = estimation("curve", cat(3, X, R), Yp, s.domain, r.eta, n_blocks, chol_tol, s.x_grid, A, s.x_piv);
        res = inference("regional", f, regions, n_points, B, 1, states);
    end
    res.d = r.d; res.eta = r.eta; res.lambda_hat = f.lambda_hat;
    res.effdim = s.effdim_path(s.eta_grid == r.eta);        % attained effective dimension (saved d = 5 path)
    res.time = toc;
    save(run_file(r), "res");
    fprintf("%s d = %d: eta = 1e%+.1f, N = %.2f (%.0f s)\n", r.name, r.d, log10(r.eta), res.effdim, res.time);
end

%% Table F.1 from the saved results: p-values in the 5-8 month region (region 2)
L = @(r) load(run_file(r), "res");
fprintf("\nPanel A: unconditional (E3)\n%-8s %-10s %6s %10s %10s\n", "d", "lambda", "N", "Increasing", "Decreasing");
rowA = @(d, eta, N, p) fprintf("%-8d 10^%-7.1f %6.2f %10.3f %10.3f\n", d, log10(eta), N, p(2, 1), p(2, 2));
S = L(runs(1)); rowA(3, S.res.eta, S.res.effdim, S.res.p_value);
rowA(5, s3.eta_hat, s3.effdim_hat, t3.p_value);
S = L(runs(2)); rowA(7, S.res.eta, S.res.effdim, S.res.p_value);
fprintf("\nPanel B: conditional (E4), Inc./Dec. at r = %s\n", strjoin(compose("%.2f", states), ", "));
rowB = @(d, eta, N, p) fprintf("%-8s 10^%-7.1f %6.2f   %s\n", d, log10(eta), N, ...
    strjoin(compose("%.3f/%.3f", squeeze(p(2, 1, :)), squeeze(p(2, 2, :))), "   "));
S = L(runs(3)); rowB("3", S.res.eta, S.res.effdim, S.res.p_value);
rowB("5 and 7", e4.eta_hat, e4.effdim_hat, e4_tests.p_value);
