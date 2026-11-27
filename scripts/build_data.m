%% Build the processed inputs of E1-E4 from the raw licensed data (README, "Data"); existing files are kept.
%   Creates data/raw/jkp/, data/raw/crsp/ and data/processed/ if needed, then writes
%   data/processed/pt_portfolios.mat:   E2 Patton-Timmermann characteristic portfolios (build_pt_portfolios)
%   data/processed/treasury_panel.mat:  E3/E4 Treasury-bill panel (build_treasury_panel)
%   data/processed/beta_decile_panel_1975_2025.mat: E1 beta-decile panel, 611 formation months (build_beta_panel)
%   results/e2_distribution.mat:        pooled quantiles of the E2 decile characteristics (display ranges in
%                                       make_figures.m)
%   The E1 panel requires JKP files with the `id` column (README, "Data vintage"); otherwise it is not saved, a
%   warning is issued, and the other inputs are still built.
root = fileparts(fileparts(mfilename("fullpath")));          % repository root (this file is in scripts/)
addpath(fullfile(root, "src"));
raw_jkp = fullfile(root, "data", "raw", "jkp"); raw_crsp = fullfile(root, "data", "raw", "crsp");
processed = fullfile(root, "data", "processed");
for folder = [raw_jkp, raw_crsp, processed], if ~isfolder(folder), mkdir(folder); end, end
need = strings(0);                                           % raw files needed for the panels not yet built
if ~isfile(fullfile(processed, "pt_portfolios.mat")), need = fullfile(raw_jkp, "jkp_global_factor_usa_" + (1964:2025) + ".csv"); end
if ~isfile(fullfile(processed, "treasury_panel.mat"))
    need = [need, fullfile(raw_crsp, ["crsp_treasury_monthly_1964_2025.csv", "crsp_riskfree_monthly_1964_2025.csv"])];
end
missing = need(~isfile(need));
if ~isempty(missing)
    error("build_data: %d raw data file(s) missing, for example %s. Place the files as described in README.md (Data).", ...
        numel(missing), missing(1));
end

panel_file = fullfile(root, "data", "processed", "pt_portfolios.mat");
if ~isfile(panel_file)
    tic; pt = panels("pt", root, 1964:2025); fprintf("E2 portfolios built in %.0f s\n", toc);
    save(panel_file, "pt");
end
load(panel_file, "pt");

panel_file = fullfile(root, "data", "processed", "treasury_panel.mat");
if ~isfile(panel_file)
    tp = panels("treasury", root);
    save(panel_file, "tp");
end

panel_file = fullfile(root, "data", "processed", "beta_decile_panel_1975_2025.mat");
if ~isfile(panel_file)                                       % formation months 1975-01 to 2025-11
    months = dateshift(datetime(1975, 1, 1):calmonths(1):datetime(2025, 11, 1), "end", "month")';
    try
        tic; panel = panels("beta_rebuild", raw_jkp, months);
        save(panel_file, "panel"); fprintf("E1 panel built in %.0f s\n", toc);
    catch err                                                % E2-E4 inputs are still built
        warning("build_data:e1", "E1 panel not built: %s (README.md, Data vintage)", err.message);
    end
end

%% E2 characteristic distribution (portfolio-month level)
summary_file = fullfile(root, "results", "e2_distribution.mat");
if isfile(summary_file), return, end
K = numel(pt.names);
p = [0 0.001 0.01 0.05 0.25 0.5 0.75 0.95 0.99 0.999 1];
summary = table();
for k = 1:K
    xk = pt.x(:, :, k); nk = pt.n(:, :, k);
    full = all(isfinite(xk), 2);
    months = pt.formation_month(full);
    q = quantile(xk(full, :), p, "all");
    summary = [summary; table(pt.names(k), pt.columns(k), months(1), months(end), sum(full), ...
        nnz(isfinite(xk)), min(nk(full, :), [], "all"), median(sum(nk(full, :), 2)), q(:)', ...
        'VariableNames', ["characteristic", "column", "first_month", "last_month", "months_all_10", ...
        "portfolio_months", "min_stocks", "median_stocks", "quantiles"])]; %#ok<AGROW>
end
if ~isfolder(fileparts(summary_file)), mkdir(fileparts(summary_file)); end
save(summary_file, "summary", "p");
