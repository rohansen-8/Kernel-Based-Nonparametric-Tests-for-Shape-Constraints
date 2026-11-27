%% Monte Carlo study (README): population shape labels, then resumable replications.
%   Optional settings, set before running: n_rep and chunk (replications per T and per checkpoint) and worker = [k K]
%   (this process computes checkpoints k, k+K, ...). Parallel runs use K independent single-thread MATLAB processes
%   launched a few seconds apart (no Parallel Computing Toolbox). The population labels are computed once into
%   results/mc_population.mat and loaded afterwards. Checkpoints go to the working folder results/mc_chunks/ and are
%   reused only if their fingerprint and replication indices match; otherwise the run stops, so a completed checkpoint
%   is never overwritten. summarize_monte_carlo.m combines them into results/mc_T<T>.mat and summarizes.
root = fileparts(fileparts(mfilename("fullpath")));          % repository root (this file is in scripts/)
addpath(fullfile(root, "src"));
if ~exist("n_rep", "var"), n_rep = 1000; end                % replications per sample size
if ~exist("chunk", "var"), chunk = 50; end                  % replications per checkpoint file
if ~exist("worker", "var"), worker = [1 1]; end            % [k K]: this process computes chunks k, k+K, ...
maxNumCompThreads(1);
mc_dir = fullfile(root, "results");

%% Specification (a new struct, so that no variable already in the workspace can enter the fingerprint)
spec = struct();
spec.T = [100 500 1000 5000 10000];
spec.rho_x = 0.8;                                           % characteristic persistence
spec.rho_R = 0.5;                                           % corr(R_1, R_2) of the bounded returns
spec.rho_v = 2 * sin(pi * spec.rho_R / 6);                  % latent Gaussian correlation giving rho_R
spec.h0 = @(x) 1 + 4 * x .* (1 - x);
spec.var_Jh0 = 2 * 43 / 15 + 2 * spec.rho_R * 25 / 9;       % Var(J h0) = 2 E h0^2 + 2 rho_R (E h0)^2
spec.sigma_eps = sqrt(spec.var_Jh0);                        % population R^2 = 1/2
spec.regions = [0.05 0.35; 0.65 0.95];
spec.n_points = 5;
spec.restrictions = ["increasing", "decreasing", "convex", "concave"];
spec.eta_grid = 10.^(-5:0.5:0);                             % penalty grid (as in E1-E4)
spec.n_blocks = 4;
spec.chol_tol = 1e-10;
spec.B = 10000;
spec.seed = 20261025;
spec.x_display = linspace(0, 1, 101)';
spec.n_nodes = 200;                                         % quadrature for population objects and payoff-norm errors
spec.effdim_target = 5;                                     % effective-dimension rule (as in E1-E4)

%% Population shape labels of h_lambda over the penalty grid and the lengthscale range at T = 100 (deterministic targets)
xi = [linspace(spec.regions(1, 1), spec.regions(1, 2), spec.n_points), ...
    linspace(spec.regions(2, 1), spec.regions(2, 2), spec.n_points)]';
pop_file = fullfile(mc_dir, "mc_population.mat");
if isfile(pop_file)
    w = warning("off", "MATLAB:dispatcher:UnresolvedFunctionHandle");   % the saved h0 handle is compared by its text
    S = load(pop_file, "spec", "pop"); pop = S.pop; warning(w);
    names = string(fieldnames(spec));
    same = arrayfun(@(f) isfield(S.spec, f) && isequal(as_value(S.spec.(f)), as_value(spec.(f))), names);
    if ~all(same)
        error("mc_population.mat was computed for a different specification (%s). Move it aside to recompute.", ...
            strjoin(names(~same), ", "));
    end
elseif worker(2) > 1
    error("mc_population.mat is missing: run once with n_rep = 0 before launching the worker processes.");
else
    band = RandStream("mrg32k3a", "Seed", spec.seed); band.Substream = 1e8;
    ell_hat = zeros(500, 1);
    for b = 1:numel(ell_hat)
        X = monte_carlo("simulate", spec.T(1), spec.rho_x, spec.rho_v, spec.sigma_eps, band);
        ell_hat(b) = kernels("median_distance", X(:));
    end
    ell_band = [quantile_sorted(ell_hat, 0.01), 1 - 1 / sqrt(2), quantile_sorted(ell_hat, 0.99)];
    labels = table();
    h0_signed = @(x) [4 - 8 * x, -(4 - 8 * x), -8 * ones(size(x)), 8 * ones(size(x))];
    for ell = ell_band
        P = monte_carlo("population", spec.h0, ell, spec.rho_R, spec.eta_grid, xi, spec.n_nodes, spec.x_display);
        P2 = monte_carlo("population", spec.h0, ell, spec.rho_R, spec.eta_grid, xi, 2 * spec.n_nodes, spec.x_display);
        quad_err = max(abs([P.dh; P.d2h] - [P2.dh; P2.d2h]), [], "all");
        signed = {P.dh, -P.dh, P.d2h, -P.d2h};
        for r = 1:2
            idx = (r - 1) * spec.n_points + (1:spec.n_points);
            s0 = h0_signed(xi(idx));
            for j = 1:4
                margin = min(signed{j}(idx, :), [], 1); margin0 = min(s0(:, j));
                truth = margin0 > 0;
                kept = all((margin > 100 * quad_err) == truth & (margin < -100 * quad_err) == ~truth);
                labels = [labels; table(ell, r, spec.restrictions(j), truth, margin0, margin, kept, quad_err, ...
                    'VariableNames', ["ell", "region", "restriction", "true_for_h0", "margin_h0", "margin", "label_kept", "quad_err"])]; %#ok<AGROW>
            end
        end
    end
    pop = struct("nodes", P.x, "weights", P.w);
    if ~isfolder(mc_dir), mkdir(mc_dir); end
    save(pop_file, "spec", "pop", "labels", "ell_hat", "ell_band", "xi");
    writetable(splitvars(labels), fullfile(mc_dir, "mc_population_labels.csv"));
    fprintf("population labels kept in all %d cells: %d\n", height(labels), all(labels.label_kept));
end

%% Replications: checkpoints of `chunk` replications per sample size, validated by a specification fingerprint
fingerprint = spec_fingerprint(spec, pop, root);
if n_rep == 0, return, end                                  % population labels only
chunk_dir = fullfile(mc_dir, "mc_chunks");                  % working folder of a run in progress
if ~isfolder(chunk_dir), mkdir(chunk_dir); end
jobs = [];
for iT = numel(spec.T):-1:1                                 % largest T first, so heavy chunks spread over processes
    for first = 1:chunk:n_rep
        jobs = [jobs; iT, first, min(first + chunk - 1, n_rep)]; %#ok<AGROW>
    end
end
files = fullfile(chunk_dir, compose("T%05d_rep%05d.mat", spec.T(jobs(:, 1))', jobs(:, 2)));
indices = arrayfun(@(j) jobs(j, 1) * 1e5 + (jobs(j, 2):jobs(j, 3)), (1:size(jobs, 1))', "UniformOutput", false);
check_checkpoints(chunk_dir, files, indices, fingerprint);
mine = worker(1):worker(2):size(jobs, 1);
status = strings(numel(mine), 1);
t_start = tic;
for i = 1:numel(mine)
    j = mine(i);
    status(i) = monte_carlo("chunk", files(j), fingerprint, spec.T(jobs(j, 1)), indices{j}, spec, pop);
end
fprintf("worker %d/%d: %d chunks (%d computed, %d reused) in %.0f s\n", worker, numel(status), nnz(status == "computed"), ...
    nnz(status == "reused"), toc(t_start));

function q = quantile_sorted(v, p)
%% Empirical quantile (type 7) without toolboxes.
v = sort(v(:)); h = (numel(v) - 1) * p + 1;
q = v(floor(h)) + (h - floor(h)) * (v(min(ceil(h), numel(v))) - v(floor(h)));
end

function check_checkpoints(chunk_dir, files, indices, fingerprint)
%% Stop before any computation if the folder holds a checkpoint this run would not produce, or one made with another
%  fingerprint or other replication indices. Completed checkpoints are therefore never recomputed or overwritten.
d = dir(fullfile(chunk_dir, "*.mat")); d = d(~endsWith({d.name}, ".tmp.mat"));
present = fullfile(chunk_dir, string({d.name}'));
if any(~ismember(present, files))
    error("%s holds checkpoints from a different n_rep or chunk size; move them aside first.", chunk_dir);
end
for i = find(ismember(files, present))'
    S = load(files(i), "fingerprint", "indices");
    if S.fingerprint ~= fingerprint || ~isequal(S.indices, indices{i})
        error("%s was computed with another specification or source (fingerprint %s, now %s); move it aside first.", ...
            files(i), S.fingerprint, fingerprint);
    end
end
end

function v = as_value(v)
%% Comparable value of a specification field (function handles by their text).
if isa(v, "function_handle"), v = func2str(v); end
end

function fp = spec_fingerprint(spec, pop, root)
%% MD5 of the specification, the quadrature used for errors, and the source of the modules the replications use.
s = spec; s.h0 = func2str(spec.h0);
text = string(jsonencode(s)) + string(jsonencode(pop));
for f = ["monte_carlo", "estimation", "kernels", "inference"]
    text = text + string(fileread(fullfile(root, "src", f + ".m")));
end
md = java.security.MessageDigest.getInstance("MD5");
md.update(unicode2native(char(text), "UTF-8"));
fp = string(lower(reshape(dec2hex(typecast(md.digest(), "uint8"))', 1, [])));
end
