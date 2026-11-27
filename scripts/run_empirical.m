%% Empirical applications E1-E4: fits and regional shape tests, saved in results/ for make_figures.m and make_tables.m.
%   The penalty is chosen by the effective-dimension rule (target 5 on the grid 10.^(-5:0.5:0); README).
%   Optional settings, set before running:
%     apps           applications to run, a subset of ["e1" "e2" "e3" "e4"] (default: all)
%     e4_subsamples  E4 subsamples to run, indices into subsamples (default 1:3)
%   Completed result files are reused and never overwritten (E1 is complete once e1_tests exists).
root = fileparts(fileparts(mfilename("fullpath")));          % repository root (this file is in scripts/)
addpath(fullfile(root, "src"));
if ~exist("apps", "var"), apps = ["e1" "e2" "e3" "e4"]; end
apps = string(apps);
if ~exist("e4_subsamples", "var"), e4_subsamples = 1:3; end
e4_run = e4_subsamples(:)'; if ~any(apps == "e4"), e4_run = []; end   % E4 subsamples this run estimates
eta_grid = 10.^(-5:0.5:0);                              % penalty grid, lambda = eta * tr(Sigma)
effdim_target = 5;                                      % target effective dimension
effdim_fields = ["effdim_target", "effdim_hat", "effdim_path"];   % effective-dimension metadata saved with each fit
maturity_domain = [1 12]; maturity_regions = [1 5; 5 8; 8 12];   % Treasury maturity domain and test regions, months
results_dir = fullfile(root, "results");
result_file = @(name) fullfile(results_dir, name + ".mat");     % e.g. results/e1_fit.mat
if ~isfolder(results_dir), mkdir(results_dir); end

n_blocks = 4;                   % argument of estimation("curve"); not used by the effective-dimension rule
chol_tol = 1e-10;               % max residual diagonal of the pivoted Cholesky basis
B = 10000;                      % Gaussian draws for the vertex calibration of the shape tests
n_points = 5;                   % derivative points per region (E2-E4)

%% E1: market beta and expected excess returns (the tests below use this fit in memory)
if any(apps == "e1") && ~isfile(result_file("e1_tests"))
    assert(~isfile(result_file("e1_fit")), "%s exists without e1_tests; move it aside to re-estimate E1.", result_file("e1_fit"));
    panel = panels("beta", root);
    domain = [-0.5 3.5];                                   % learning domain
    f = estimation("curve", panel.x, panel.Y, domain, eta_grid, n_blocks, chol_tol, linspace(domain(1), domain(2), 401)', ...
        [], [], "target_effdim", effdim_target);
    [L, piv, x_piv, ell, x, y, w, t, a, beta, lambda_hat, eta_hat, T] = ...
        deal(f.L, f.piv, f.x_piv, f.ell, f.x, f.y, f.w, f.t, f.a, f.beta, f.lambda_hat, f.eta_hat, f.T);
    e1 = struct("domain", domain, "ell", ell, "eta_grid", eta_grid, "eta_hat", eta_hat, "lambda_hat", lambda_hat, ...
        "trace_Sigma", lambda_hat / eta_hat, "a", a, "c", f.c, "x_piv", x_piv, "rank", numel(piv), "chol_tol", chol_tol, ...
        "n_obs", numel(x), "n_months", size(panel.x, 1), "x_grid", f.x_grid, "h_grid", f.h_grid, "h_path", f.h_path, ...
        "decile_mean_x", mean(panel.x)', "decile_mean_y", mean(panel.Y)', "time_chol", f.time_chol);
    for field = effdim_fields, e1.(field) = f.(field); end
    save(result_file("e1_fit"), "e1");
    fprintf("E1: n = %d, ell = %.4f, rank = %d (%.1f s), eta_hat = %.3g, lambda_hat = %.3g, effective dimension %.2f\n", ...
        numel(x), ell, numel(piv), f.time_chol, eta_hat, lambda_hat, f.effdim_hat);
end

%% E1: regional shape tests (weakly increasing and weakly decreasing in each region)
if any(apps == "e1") && ~isfile(result_file("e1_tests"))
    regions = [0 0.75; 0.75 1.25; 1.25 2.75];             % low, market, high beta
    xi = (0:0.25:2.75)';                                   % common derivative grid, spacing 0.25 (about 0.4 ell)
    D = kernels("matern52_dx", xi, x_piv, ell) / L(piv, :)';          % h'(xi) = D * beta
    theta_hat = D * beta;
    G = inference("scores", L, y, w, t, a, beta, lambda_hat, D);   % T x n monthly grid scores
    n_lags = floor(4 * (T / 100)^(2/9));
    Omega = inference("newey_west", G, n_lags);
    rng(1, "twister");
    E = randn(B, numel(xi));                               % common calibration draws
    n_regions = size(regions, 1);
    W = zeros(n_regions, 2); p_value = zeros(n_regions, 2); cond_Omega = zeros(n_regions, 1);
    W_null = cell(n_regions, 1);
    for r = 1:n_regions
        idx = xi >= regions(r, 1) & xi <= regions(r, 2);
        [W(r, :), p_value(r, :), W_null{r}] = inference("shape_test", [theta_hat(idx), -theta_hat(idx)], Omega(idx, idx), T, E(:, 1:nnz(idx)));
        cond_Omega(r) = cond(Omega(idx, idx));
    end
    e1_tests = struct("regions", regions, "xi", xi, "theta_hat", theta_hat, "G", G, "Omega", Omega, ...
        "n_lags", n_lags, "B", B, "seed", 1, "W", W, "p_value", p_value, "W_null", {W_null}, ...
        "cond_Omega", cond_Omega, "eta", eta_hat, "lambda", lambda_hat, "T", T);
    save(result_file("e1_tests"), "e1_tests");
    disp(array2table([W p_value cond_Omega], "VariableNames", ["W_increasing", "W_decreasing", "p_increasing", "p_decreasing", "cond_Omega"], ...
        "RowNames", ["Low beta", "Market beta", "High beta"]))
end

%% E2: the eight Patton-Timmermann characteristics
if any(apps == "e2") && ~isfile(result_file("e2_fit"))      % a completed fit is reused
    load(fullfile(root, "data", "processed", "pt_portfolios.mat"), "pt");
    domains = [1 1e6; 0.025 3.5; 0.003 1.1; 0.004 0.33; 0.002 0.12; -0.85 2.5; -0.4 0.5; -0.95 8];  % learning domains (original units)
    e2 = struct([]);
    for k = 1:numel(pt.names)
        x_panel = pt.x(:, :, k);
        domain_input = domains(k, :);
        input = pt.columns(k);
        if k == 1                                            % size: natural log of portfolio market equity
            x_panel = log(x_panel);
            domain_input = log(domain_input);
            input = "log(me)";
        end
        x_grid = linspace(domain_input(1), domain_input(2), 401)';
        f = estimation("curve", x_panel, pt.Y(:, :, k), domain_input, eta_grid, n_blocks, chol_tol, x_grid, [], [], ...
            "target_effdim", effdim_target);
        e2(k).name = pt.names(k); e2(k).column = pt.columns(k); e2(k).input = input; e2(k).direction = pt.direction(k);
        e2(k).domain = domains(k, :); e2(k).domain_input = domain_input; e2(k).eta_grid = eta_grid;
        for field = ["ell", "eta_hat", "lambda_hat", "a", "c", "x_piv", "x_grid", "h_grid", "h_path", "time_chol", ...
                "T", effdim_fields]
            e2(k).(field) = f.(field);
        end
        e2(k).rank = numel(f.piv); e2(k).n_obs = numel(f.x); e2(k).trace_Sigma = f.lambda_hat / f.eta_hat;
        e2(k).decile_mean_x = mean(x_panel, "omitnan")';    % in regression-input units
        e2(k).decile_mean_y = mean(pt.Y(:, :, k), "omitnan")';
        fprintf("E2 %-20s n = %d, ell = %.4g, rank = %d (%.1f s), eta_hat = %.3g, effective dimension %.2f\n", ...
            pt.names(k), numel(f.x), f.ell, numel(f.piv), f.time_chol, f.eta_hat, f.effdim_hat);
    end
    save(result_file("e2_fit"), "e2");
end

%% E2: regional shape tests at the fitted eta (rebuilds each basis from the saved pivots)
if any(apps == "e2") && ~isfile(result_file("e2_tests"))    % completed tests are reused
    load(result_file("e2_fit"), "e2");
    regions_e2 = {[5 100; 100 1e4; 1e4 2e5], [0.10 0.60; 0.60 1.20; 1.20 2.20], [0.01 0.10; 0.10 0.25; 0.25 0.40], ...
        [0.01 0.05; 0.05 0.10; 0.10 0.20], [0.004 0.02; 0.02 0.05; 0.05 0.08], [-0.60 0; 0 0.50; 0.50 1.10], ...
        [-0.20 -0.05; -0.05 0.05; 0.05 0.25], [-0.75 0; 0 1; 1 3.50]};    % original units
    e2_tests = struct([]);
    for k = 1:numel(e2)
        x_panel = pt.x(:, :, k);
        regions = regions_e2{k};
        if e2(k).input == "log(me)", x_panel = log(x_panel); regions = log(regions); end
        f = estimation("curve", x_panel, pt.Y(:, :, k), e2(k).domain_input, e2(k).eta_hat, n_blocks, chol_tol, e2(k).x_grid, [], e2(k).x_piv);
        out = inference("regional", f, regions, n_points, B, 1);
        out.name = e2(k).name; out.input = e2(k).input; out.regions_original = regions_e2{k};
        out.max_diff_saved_fit = max(abs(f.h_grid - e2(k).h_grid));      % rebuilt fit equals the saved fit
        e2_tests = [e2_tests, out]; %#ok<AGROW>
        fprintf("E2 tests %-20s p(inc) %s  p(dec) %s  cond %s\n", e2(k).name, mat2str(out.p_value(:, 1)', 3), ...
            mat2str(out.p_value(:, 2)', 3), mat2str(out.cond_Omega', 3));
    end
    save(result_file("e2_tests"), "e2_tests");
end

%% Treasury applications (E3, E4): panel and subsamples
if any(apps == "e3" | apps == "e4")
    load(fullfile(root, "data", "processed", "treasury_panel.mat"), "tp");
    subsamples = [1964 1972; 1973 2001; 2009 2025];
end

%% E3: unconditional Treasury-bill term premia (expected excess return on remaining maturity), three subsamples
if any(apps == "e3") && ~isfile(result_file("e3_fit"))      % a completed fit is reused
    domain = maturity_domain;                          % maturity learning domain, months
    buckets = floor(domain(1)):ceil(domain(2));        % one-month maturity buckets for reference means
    x_grid = linspace(domain(1), domain(2), round(diff(domain) / 0.0225) + 1)';   % 0.0225-month steps
    e3 = struct([]);
    for s = 1:size(subsamples, 1)
        d = tp(year(tp.formation_date) >= subsamples(s, 1) & year(tp.formation_date) <= subsamples(s, 2), :);
        [X, Yp, A, months] = panels("treasury_month", d);
        f = estimation("curve", X, Yp, domain, eta_grid, n_blocks, chol_tol, x_grid, A, [], "target_effdim", effdim_target);
        bucket = discretize(f.x, buckets);                 % amount-weighted bucket mean per month, averaged over months
        bucket_mean = zeros(numel(buckets) - 1, 1);
        for b = 1:numel(buckets) - 1
            in = bucket == b;
            bucket_mean(b) = mean(accumarray(f.t(in), f.w(in) .* f.y(in), [], [], NaN) ./ accumarray(f.t(in), f.w(in), [], [], NaN), "omitnan");
        end
        e3(s).subsample = subsamples(s, :); e3(s).months = [months(1) months(end)]; e3(s).domain = domain;
        e3(s).eta_grid = eta_grid; e3(s).weights = "amount outstanding, normalized within month";
        for field = ["ell", "eta_hat", "lambda_hat", "a", "c", "x_piv", "x_grid", "h_grid", "h_path", "time_chol", ...
                "T", effdim_fields]
            e3(s).(field) = f.(field);
        end
        e3(s).rank = numel(f.piv); e3(s).n_obs = numel(f.x); e3(s).trace_Sigma = f.lambda_hat / f.eta_hat;
        e3(s).bucket_mid = (buckets(1:end - 1) + 0.5)'; e3(s).bucket_mean = bucket_mean;
        fprintf("E3 %d-%d: n = %d, T = %d, ell = %.3f, rank = %d (%.1f s), eta_hat = %.3g, effective dimension %.2f\n", ...
            subsamples(s, :), numel(f.x), f.T, f.ell, numel(f.piv), f.time_chol, f.eta_hat, f.effdim_hat);
    end
    save(result_file("e3_fit"), "e3");
end

%% E3: regional shape tests at the fitted eta (rebuilds each basis from the saved pivots)
if any(apps == "e3") && ~isfile(result_file("e3_tests"))    % completed tests are reused
    load(result_file("e3_fit"), "e3");
    regions = maturity_regions;                        % common maturity regions, months
    e3_tests = struct([]);
    for s = 1:numel(e3)
        assert(isequal(e3(s).domain, maturity_domain), "saved E3 fit has another maturity domain");
        d = tp(year(tp.formation_date) >= e3(s).subsample(1) & year(tp.formation_date) <= e3(s).subsample(2), :);
        [X, Yp, A] = panels("treasury_month", d);
        f = estimation("curve", X, Yp, e3(s).domain, e3(s).eta_hat, n_blocks, chol_tol, e3(s).x_grid, A, e3(s).x_piv);
        out = inference("regional", f, regions, n_points, B, 1);
        out.subsample = e3(s).subsample;
        out.max_diff_saved_fit = max(abs(f.h_grid - e3(s).h_grid));      % rebuilt fit equals the saved fit
        e3_tests = [e3_tests, out]; %#ok<AGROW>
        fprintf("E3 tests %d-%d  p(inc) %s  p(dec) %s  cond %s\n", e3(s).subsample, mat2str(out.p_value(:, 1)', 3), ...
            mat2str(out.p_value(:, 2)', 3), mat2str(out.cond_Omega', 3));
    end
    save(result_file("e3_tests"), "e3_tests");
end

%% E4: conditional Treasury-bill term premia (maturity and short rate), three subsamples; one result file each
for s = e4_run
    file = result_file(sprintf("e4_%d_%d_fit", subsamples(s, :)));
    if isfile(file), continue, end                    % completed subsamples are reused
    d = tp(year(tp.formation_date) >= subsamples(s, 1) & year(tp.formation_date) <= subsamples(s, 2), :);
    [X, Yp, A, months, R] = panels("treasury_month", d);
    domain = [maturity_domain; min(d.r) max(d.r)];    % maturity (months) x short rate (% p.a.)
    [M, Rg] = ndgrid(linspace(domain(1, 1), domain(1, 2), 10 * diff(domain(1, :)) + 1), ...   % 0.1-month steps
        linspace(domain(2, 1), domain(2, 2), 50));
    f = estimation("curve", cat(3, X, R), Yp, domain, eta_grid, n_blocks, chol_tol, [M(:) Rg(:)], A, [], "target_effdim", effdim_target);
    e4 = struct("subsample", subsamples(s, :), "months", [months(1) months(end)], "domain", domain, ...
        "eta_grid", eta_grid, "ell", f.ell, "eta_hat", f.eta_hat, "lambda_hat", f.lambda_hat, ...
        "a", f.a, "c", f.c, "x_piv", f.x_piv, ...
        "x_grid", f.x_grid, "grid_size", size(M), "h_grid", f.h_grid, "h_path", f.h_path, "rank", numel(f.piv), ...
        "n_obs", size(f.x, 1), "T", f.T, "trace_Sigma", f.lambda_hat / f.eta_hat, ...
        "time_chol", f.time_chol, "time_fit", f.time_fit);
    for field = effdim_fields, e4.(field) = f.(field); end
    save(file, "e4");
    fprintf("E4 %d-%d: n = %d, ell = %s, rank = %d (%.0f s), eta_hat = %.3g, effective dimension %.2f\n", subsamples(s, :), ...
        size(f.x, 1), mat2str(f.ell, 3), numel(f.piv), f.time_chol, f.eta_hat, f.effdim_hat);
end

%% E4: regional shape tests in maturity at fixed short rates (rebuilds each saved basis from its pivots; no refit)
regions = maturity_regions;                           % E3 maturity regions, months
states = {[3.40 4.43 6.31], [3.64 5.73 9.56], [0.14 4.43]};   % short-rate levels, % p.a.
for s = e4_run
    file = result_file(sprintf("e4_%d_%d_tests", subsamples(s, :)));
    if isfile(file), continue, end                    % completed subsamples are reused
    load(result_file(sprintf("e4_%d_%d_fit", subsamples(s, :))), "e4");
    assert(isequal(e4.domain(1, :), maturity_domain), "saved E4 fit has another maturity domain");
    d = tp(year(tp.formation_date) >= subsamples(s, 1) & year(tp.formation_date) <= subsamples(s, 2), :);
    [X, Yp, A, ~, R] = panels("treasury_month", d);
    f = estimation("curve", cat(3, X, R), Yp, e4.domain, e4.eta_hat, n_blocks, chol_tol, e4.x_grid, A, e4.x_piv);
    tic; e4_tests = inference("regional", f, regions, n_points, B, 1, states{s}); e4_tests.time_tests = toc;
    e4_tests.subsample = subsamples(s, :); e4_tests.ell = f.ell; e4_tests.rank = numel(f.piv);
    e4_tests.time_basis = f.time_chol; e4_tests.time_fit = f.time_fit;
    e4_tests.max_diff_saved_fit = max(abs(f.h_grid - e4.h_grid));  % rebuilt fit equals the saved fit
    save(file, "e4_tests");
    fprintf("E4 tests %d-%d: basis %.0f s, fit %.0f s, tests %.0f s, max |h - saved| %.1e, lag %d\n", subsamples(s, :), ...
        f.time_chol, f.time_fit, e4_tests.time_tests, e4_tests.max_diff_saved_fit, e4_tests.n_lags);
end
