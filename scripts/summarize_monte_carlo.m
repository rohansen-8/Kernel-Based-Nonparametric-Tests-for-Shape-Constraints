%% Numerical summaries of the Monte Carlo study from the saved replications only (no re-estimation).
%   Completeness and random-number streams; rejection frequencies against T for each restriction and region (Figure
%   G.2); estimation errors at every T (Table G.1), with the replication-level errors at T = 1000 saved for Figure G.1;
%   selected penalties and diagnostics. After a run, the checkpoints in results/mc_chunks/ are first combined into
%   results/mc_T<T>.mat, one file per sample size (existing files are never overwritten). Writes
%   results/mc_{completeness,rejection_frequencies,estimation_errors,eta_frequencies,diagnostics}.csv and
%   results/mc_summary.mat.
root = fileparts(fileparts(mfilename("fullpath")));          % repository root (this file is in scripts/)
mc_dir = fullfile(root, "results");
out_dir = mc_dir;
w = warning("off", "MATLAB:dispatcher:UnresolvedFunctionHandle");      % the saved h0 handle evaluates without its source file
P = load(fullfile(mc_dir, "mc_population.mat"), "spec", "labels"); spec = P.spec; warning(w); alpha = [0.01 0.05 0.10];
rep_file = @(T) fullfile(mc_dir, sprintf("mc_T%05d.mat", T));

%% Combine the checkpoints of a new run (results/mc_chunks/) into one file per sample size
chunk_dir = fullfile(mc_dir, "mc_chunks");
if isfolder(chunk_dir)
    for iT = 1:numel(spec.T)
        T = spec.T(iT); assert(~isfile(rep_file(T)), "%s exists; move it aside before combining a new run.", rep_file(T));
        files = dir(fullfile(chunk_dir, sprintf("T%05d_rep*.mat", T)));
        files = files(~endsWith({files.name}, ".tmp.mat"));     % ignore interrupted temporary saves
        [~, order] = sort({files.name}); parts = arrayfun(@(d) load(fullfile(d.folder, d.name)), files(order));
        fingerprint = unique(string({parts.fingerprint})); assert(isscalar(fingerprint), "mixed fingerprints for T = %d", T);
        results = [parts.results]; indices = [parts.indices]; chunks = numel(parts);
        save(rep_file(T), "results", "indices", "T", "fingerprint", "chunks");
    end
end

%% Completeness and random-number streams
R = cell(numel(spec.T), 1); fps = strings(0); checks = table();
for iT = 1:numel(spec.T)
    C = load(rep_file(spec.T(iT)));
    fps = [fps; C.fingerprint]; %#ok<AGROW>
    idx = C.indices; R{iT} = C.results;
    checks = [checks; table(spec.T(iT), C.chunks, numel(R{iT}), numel(unique(idx)), ...
        isequal(sort(idx), iT * 1e5 + (1:numel(idx))), 'VariableNames', ["T", "chunks", "replications", ...
        "unique_streams", "streams_contiguous"])]; %#ok<AGROW>
end
checks.fingerprints_identical = repmat(numel(unique(fps)) == 1, height(checks), 1);

%% Rejection frequency (p <= alpha) against T
rej = table();
for iT = 1:numel(spec.T)
    p = cat(3, R{iT}.p);                                    % region x restriction x replication
    n = size(p, 3);
    for r = 1:2
        for j = 1:4
            f = arrayfun(@(a) mean(p(r, j, :) <= a), alpha);
            truth = P.labels.true_for_h0(find(P.labels.region == r & P.labels.restriction == spec.restrictions(j), 1));
            kept = all(P.labels.label_kept(P.labels.region == r & P.labels.restriction == spec.restrictions(j)));
            rej = [rej; table(spec.T(iT), r, spec.restrictions(j), truth, kept, n, f(1), f(2), f(3), ...
                sqrt(f(2) * (1 - f(2)) / n), 'VariableNames', ["T", "region", "restriction", "restriction_true", ...
                "population_label_kept", "n", "rate_1pct", "rate_5pct", "rate_10pct", "se_5pct"])]; %#ok<AGROW>
        end
    end
end

%% Estimation errors (payoff norm and pointwise h_hat(x) - h0(x)); replication-level data at T = 1000
q = [0.05 0.25 0.5 0.75 0.95]; xd = spec.x_display; h0d = spec.h0(xd);
learn = table();
for iT = 1:numel(spec.T)
    e = [R{iT}.err]'; dev = [R{iT}.h] - h0d;               % display point x replication
    in = xd >= 0.05 & xd <= 0.95;
    learn = [learn; table(spec.T(iT), mean(e), quantile7(e, q), mean(sqrt(mean(dev(in, :).^2, 1))), ...
        'VariableNames', ["T", "mean_payoff_error", "payoff_error_quantiles", "mean_rmse_pointwise_0p05_0p95"])]; %#ok<AGROW>
end
iB = find(spec.T == 1000);
errors_T1000 = struct("x", xd, "pointwise_error", [R{iB}.h] - h0d, "payoff_error", [R{iB}.err]');

%% Selected penalties and diagnostics
eta_freq = table(); diag_t = table();
for iT = 1:numel(spec.T)
    eh = [R{iT}.eta_hat]'; ell = [R{iT}.ell]'; rk = [R{iT}.rank]';
    eta_freq = [eta_freq; table(spec.T(iT), arrayfun(@(v) nnz(eh == v), spec.eta_grid), 'VariableNames', ["T", "eta_counts"])]; %#ok<AGROW>
    bad = arrayfun(@(r) any(~isfinite([r.p(:); r.W(:); r.err])), R{iT});
    diag_t = [diag_t; table(spec.T(iT), quantile7(ell, [0.01 0.5 0.99]), unique([R{iT}.n_lags]), mean(rk), [min(rk), max(rk)], ...
        mean([R{iT}.time_total]), nnz(bad), 'VariableNames', ["T", "ell_hat_q01_q50_q99", "hac_lags", "mean_rank", ...
        "rank_range", "mean_seconds", "non_finite"])]; %#ok<AGROW>
end

%% Save
writetable(checks, fullfile(out_dir, "mc_completeness.csv"));
writetable(rej, fullfile(out_dir, "mc_rejection_frequencies.csv"));
writetable(splitvars(learn), fullfile(out_dir, "mc_estimation_errors.csv"));
writetable(splitvars(eta_freq), fullfile(out_dir, "mc_eta_frequencies.csv"));
writetable(splitvars(diag_t), fullfile(out_dir, "mc_diagnostics.csv"));
save(fullfile(out_dir, "mc_summary.mat"), "checks", "rej", "learn", "errors_T1000", "eta_freq", "diag_t", "alpha");
disp(checks); disp(rej(:, ["T", "region", "restriction", "restriction_true", "rate_5pct", "se_5pct"]));

function v = quantile7(x, p)
%% Empirical quantiles (type 7) of a column vector without toolboxes.
x = sort(x(:)); n = numel(x); h = (n - 1) * p + 1; lo = floor(h); hi = min(ceil(h), n);
v = x(lo)' + (h - lo) .* (x(hi)' - x(lo)');
end
