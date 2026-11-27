%% Figures of the paper from saved results only (no estimation, inference or simulation): Figures 5.1-5.4 and G.1-G.3.
%   Empirical figures (5.1-5.4, G.3) need the results of build_data.m and run_empirical.m; Monte Carlo figures (G.1,
%   G.2) need only those of run_monte_carlo.m and summarize_monte_carlo.m, so they can be drawn without the licensed
%   data. Each group is drawn when its inputs are present. Writes vector PDFs to figures/. Figures are drawn at their printed size (\textwidth = 16.51 cm; widths below are
%   the manuscript's \includegraphics fractions), so the stated font sizes are the printed sizes.
root = fileparts(fileparts(mfilename("fullpath")));          % repository root (this file is in scripts/)
addpath(fullfile(root, "src"));                       % kernel functions to evaluate saved E4 models
results_dir = fullfile(root, "results");
fig_dir = fullfile(root, "figures");
if ~isfolder(fig_dir), mkdir(fig_dir); end
e4_files = ["e4_1964_1972_fit.mat", "e4_1973_2001_fit.mat", "e4_2009_2025_fit.mat"];
inputs_emp = [fullfile(results_dir, ["e1_fit.mat", "e2_fit.mat", "e2_distribution.mat", "e3_fit.mat", e4_files, ...
    replace(e4_files, "_fit", "_tests")]), fullfile(root, "data", "processed", "pt_portfolios.mat")];
inputs_mc = fullfile(results_dir, ["mc_summary.mat", "mc_population.mat"]);
draw_emp = all(isfile(inputs_emp)); draw_mc = all(isfile(inputs_mc));
if any(isfile(inputs_emp)) && ~draw_emp
    error("make_figures: empirical inputs missing: %s", strjoin(inputs_emp(~isfile(inputs_emp)), ", "));
end
if ~draw_emp && ~draw_mc, error("make_figures: no saved results in %s (README, Replication).", results_dir); end
if ~draw_emp, fprintf("Empirical results not found: drawing the Monte Carlo figures only.\n"); end
if ~draw_mc, fprintf("Monte Carlo summaries not found: drawing the empirical figures only.\n"); end

% Figure style: multiple curves differ in colour and line style, so they stay distinct in grayscale and for
% colour-blind readers. Sizes in points at the printed size.
st.font = "Times New Roman"; st.tick = 9; st.label = 10; st.legend = 9; st.line = 1.5;
st.small = struct("tick", 8, "label", 9, "legend", 8);   % crowded panels: the 3-D surfaces
st.red = [139 26 26] / 255; st.slate = [47 93 124] / 255; st.green = [34 85 34] / 255;
st.series = struct("color", {st.red, st.slate, st.green}, "line", {"-", "--", "-."});   % E3 subsamples; E4 short-rate states
st.levels = struct("color", {st.slate, st.red, st.green}, "line", {"-", "--", "-."}, "marker", {"o", "s", "^"});   % 1%, 5%, 10%
tw = 16.51;                                           % manuscript \textwidth, cm

if draw_emp
    %% Figure 5.1 (e1_curve): fitted expected excess return against market beta, percent per month
    load(fullfile(results_dir, "e1_fit.mat"), "e1");
    in = e1.x_grid >= 0 & e1.x_grid <= 2.75;
    [fig, t] = paper_figure(0.55 * tw, 6.4);
    ax = nexttile(t);
    plot(ax, e1.x_grid(in), 100 * e1.h_grid(in), "Color", st.red, "LineWidth", st.line);
    xlim(ax, [0 2.75]); style_axes(ax, st);
    xlabel(ax, "Market beta", "FontSize", st.label);
    ylabel(ax, "Expected excess return (% per month)", "FontSize", st.label);
    save_figure(fig, fullfile(fig_dir, "e1_curve.pdf"));

    %% Figures 5.2 (e2_profiles) and G.3 (e2_histograms): eight characteristics on the pooled 1st-99th percentile ranges
    load(fullfile(results_dir, "e2_fit.mat"), "e2");
    load(fullfile(results_dir, "e2_distribution.mat"), "summary", "p");   % data display ranges (build_data.m)
    q_lo = summary.quantiles(:, p == 0.01); q_hi = summary.quantiles(:, p == 0.99);
    labels = ["Size (USD million)", "Book-to-market", "Cash flow-to-price", "Earnings-to-price", "Dividend yield", ...
        "Momentum", "Short-term reversal", "Long-term reversal"];          % same order as e2
    show = @(k) [max(q_lo(k), e2(k).domain(1)), min(q_hi(k), e2(k).domain(2))];
    is_size = @(k) e2(k).input == "log(me)";          % size: saved grid is log market equity, plotted on a log axis

    [fig, t] = paper_figure(0.92 * tw, 19.5, 4, 2);
    for k = 1:numel(e2)
        ax = nexttile(t);
        xx = e2(k).x_grid; if is_size(k), xx = exp(xx); end
        s = show(k); in = xx >= s(1) & xx <= s(2);
        yy = 100 * e2(k).h_grid(in);
        plot(ax, xx(in), yy, "Color", st.red, "LineWidth", st.line);
        xlim(ax, s); ylim(ax, [min(yy), max(yy)] + [-0.1 0.1] * (max(yy) - min(yy)));
        style_axes(ax, st);
        if is_size(k), set(ax, "XScale", "log", "XTick", 10.^(1:5), "XMinorGrid", "off", "XMinorTick", "off"); end
        xlabel(ax, labels(k), "FontSize", st.label);
    end
    ylabel(t, "Expected excess return (% per month)", "FontName", st.font, "FontSize", st.label);
    save_figure(fig, fullfile(fig_dir, "e2_profiles.pdf"));

    %   Observations: pt.x, all months with all ten deciles (the sample of results/e2_distribution.mat). 30 equal-width
    %   bins over the display range (equal width in log size for Size); bar height = bin count / all observations.
    load(fullfile(root, "data", "processed", "pt_portfolios.mat"), "pt");
    [fig, t] = paper_figure(0.9 * tw, 19.5, 4, 2);
    for k = 1:numel(e2)
        ax = nexttile(t);
        xk = pt.x(:, :, k); v = xk(all(isfinite(xk), 2), :);
        s = show(k);
        if is_size(k), edges = logspace(log10(s(1)), log10(s(2)), 31); else, edges = linspace(s(1), s(2), 31); end
        h = histogram(ax, v(:), edges, "Normalization", "probability", "FaceColor", [178 72 72] / 255, "FaceAlpha", 1, ...
            "EdgeColor", "w", "LineWidth", 0.25);
        xlim(ax, s); ylim(ax, [0, 1.1 * max(h.Values)]);
        style_axes(ax, st);
        if is_size(k), set(ax, "XScale", "log", "XTick", 10.^(1:5), "XMinorGrid", "off", "XMinorTick", "off"); end
        xlabel(ax, labels(k), "FontSize", st.label);
    end
    ylabel(t, "Relative frequency", "FontName", st.font, "FontSize", st.label);
    save_figure(fig, fullfile(fig_dir, "e2_histograms.pdf"));

    %% Figure 5.3 (e3_profiles): fitted maturity curves of the three subsamples, bp per month
    load(fullfile(results_dir, "e3_fit.mat"), "e3");
    [fig, t] = paper_figure(0.6 * tw, 6.9);
    ax = nexttile(t); hold(ax, "on");
    for s = 1:numel(e3)
        plot(ax, e3(s).x_grid, 1e4 * e3(s).h_grid, st.series(s).line, "Color", st.series(s).color, "LineWidth", st.line);
    end
    xlim(ax, e3(1).domain); style_axes(ax, st);
    xlabel(ax, "Remaining maturity (months)", "FontSize", st.label);
    ylabel(ax, "Expected excess return (bp per month)", "FontSize", st.label);
    legend(ax, compose("%d" + char(8211) + "%d", vertcat(e3.subsample)), "FontSize", st.legend, ...
        "Orientation", "horizontal", "Location", "northoutside");   % above the axes: clear of all three curves
    save_figure(fig, fullfile(fig_dir, "e3_profiles.pdf"));

    %% Figure 5.4 (e4_profiles_surfaces): conditional curves at the tested short rates (left) and fitted surfaces (right)
    e4_tests = struct([]);
    for f = e4_files
        S = load(fullfile(results_dir, replace(f, "_fit.mat", "_tests.mat")), "e4_tests"); e4_tests = [e4_tests, S.e4_tests]; %#ok<AGROW>
    end
    rates = {e4_tests.states};                                      % tested short rates, % p.a.
    plot_maturity = @(e4) linspace(e4.domain(1, 1), e4.domain(1, 2), 20 * diff(e4.domain(1, :)) + 1)';   % 0.05-month steps
    %   Short-rate states keep one style each (10th, 50th, 90th percentile) across panels; 2009-2025 has no 10th.
    surface_colors = interp1([0 0.6 1], [222 230 238; 47 93 124; 22 40 66] / 255, linspace(0, 1, 256));  % pale slate to navy
    legend_spots = ["northeast", "best", "best"];          % 1964-1972: legend clear of the 4.43% curve
    y_limits = {[0 8], [], []};                            % 1964-1972: headroom for the legend ([] keeps the automatic limits)
    [fig, t] = paper_figure(0.9 * tw, 19.5, 3, 2, "loose");   % loose: 3-D axis labels clear of the left panels
    for s = 1:numel(e4_files)
        load(fullfile(results_dir, e4_files(s)), "e4");
        h = @(m, r) e4.a + (kernels("matern52", m, e4.x_piv(:, 1), e4.ell(1)) .* kernels("laplace", r, e4.x_piv(:, 2), e4.ell(2))) * e4.c;
        maturity = plot_maturity(e4);
        ax = nexttile(t); hold(ax, "on");
        sty = st.series(end - numel(rates{s}) + 1:end);
        for k = 1:numel(rates{s})
            plot(ax, maturity, 1e4 * h(maturity, repmat(rates{s}(k), size(maturity))), sty(k).line, ...
                "Color", sty(k).color, "LineWidth", st.line);
        end
        xlim(ax, e4.domain(1, :));
        if ~isempty(y_limits{s}), ylim(ax, y_limits{s}); end
        style_axes(ax, st);
        xlabel(ax, "Remaining maturity (months)", "FontSize", st.label);
        title(ax, sprintf("%d" + char(8211) + "%d", e4.subsample), "FontSize", st.label, "FontWeight", "normal");
        legend(ax, compose("Short rate %.2f%%", rates{s}), "FontSize", st.legend, "Location", legend_spots(s));
        m_grid = linspace(e4.domain(1, 1), e4.domain(1, 2), 51)'; r_grid = linspace(e4.domain(2, 1), e4.domain(2, 2), 41)';
        Km = kernels("matern52", m_grid, e4.x_piv(:, 1), e4.ell(1));     % kernel factors computed once per subsample
        Kr = kernels("laplace", r_grid, e4.x_piv(:, 2), e4.ell(2));
        H = zeros(numel(m_grid), numel(r_grid));
        for j = 1:numel(r_grid), H(:, j) = e4.a + (Km .* Kr(j, :)) * e4.c; end   % one short rate at a time
        ax = nexttile(t);
        surf(ax, m_grid, r_grid, 1e4 * H', "FaceColor", "flat", "EdgeColor", "flat", "LineWidth", 0.1);   % facet-coloured edges: no seams in vector output
        colormap(ax, surface_colors); view(ax, -37.5, 30);
        axis(ax, "tight"); xlim(ax, e4.domain(1, :)); ylim(ax, e4.domain(2, :));
        style_axes(ax, st); ax.FontSize = st.small.tick;
        xlabel(ax, "Maturity (months)", "FontSize", st.small.label);
        ylabel(ax, "Short rate (% p.a.)", "FontSize", st.small.label);
        zlabel(ax, "bp per month", "FontSize", st.small.label);
        title(ax, sprintf("%d" + char(8211) + "%d", e4.subsample), "FontSize", st.label, "FontWeight", "normal");
    end
    ylabel(t, "Expected excess return (bp per month)", "FontName", st.font, "FontSize", st.label);
    save_figure(fig, fullfile(fig_dir, "e4_profiles_surfaces.pdf"));
end

if draw_mc
    %% Figure G.1 (mc_pointwise_errors): pointwise errors h_hat(x) - h0(x) across the 1,000 replications at T = 1,000
    %   From mc_summary.mat (errors_T1000, saved by summarize_monte_carlo.m); boxes at x = 0, 0.1, ..., 1 of the saved
    %   display grid. Boxes: quartiles and median; whiskers: 1.5 IQR (boxchart default); dots: beyond whiskers.
    load(fullfile(results_dir, "mc_summary.mat"), "errors_T1000");
    show = ismember(round(100 * errors_T1000.x), 0:10:100);
    xs = errors_T1000.x(show); err = errors_T1000.pointwise_error(show, :);   % points x replications
    [fig, t] = paper_figure(0.65 * tw, 7.4);
    ax = nexttile(t); hold(ax, "on");
    yline(ax, 0, "--", "Color", [0.5 0.5 0.5], "LineWidth", 0.75, "Layer", "bottom");   % zero reference, behind the boxes
    blue = [70 130 180] / 255;
    boxchart(ax, repelem(xs, size(err, 2)), reshape(err', [], 1), "BoxFaceColor", [153 186 213] / 255, "BoxFaceAlpha", 1, ...
        "BoxEdgeColor", blue, "WhiskerLineColor", blue, "MarkerColor", blue, "BoxMedianLineColor", st.red, ...
        "MarkerStyle", ".", "BoxWidth", 0.06, "LineWidth", 1);           % width in x units (points 0.1 apart)
    plot(ax, xs, mean(err, 2), "o", "MarkerSize", 5, "MarkerEdgeColor", "k", "MarkerFaceColor", "none", "LineWidth", 1);   % pointwise means
    xlim(ax, [-0.06 1.06]); xticks(ax, xs); style_axes(ax, st);
    xlabel(ax, "State variable x", "FontSize", st.label);
    ylabel(ax, "Recovery error", "FontSize", st.label);
    save_figure(fig, fullfile(fig_dir, "mc_pointwise_errors.pdf"));

    %% Figure G.2 (mc_rejection_frequencies): rejection frequencies against T at the 1%, 5% and 10% levels
    %   From mc_summary.mat (rej, saved by summarize_monte_carlo.m); 1,000 replications per T. Rows: restrictions;
    %   columns: regions. Thin dashed lines: nominal levels, in the colour of their level.
    load(fullfile(results_dir, "mc_summary.mat"), "rej", "alpha");
    w = warning("off", "MATLAB:dispatcher:UnresolvedFunctionHandle");      % the saved h0 handle is not needed here
    load(fullfile(results_dir, "mc_population.mat"), "spec"); warning(w);
    rate_vars = ["rate_1pct", "rate_5pct", "rate_10pct"]; hp = gobjects(1, 3);
    [fig, t] = paper_figure(0.85 * tw, 17.5, 4, 2);
    for j = 1:numel(spec.restrictions)
        for r = 1:2
            ax = nexttile(t); hold(ax, "on");
            for a = 1:3
                yline(ax, alpha(a), "--", "Color", st.levels(a).color, "LineWidth", 0.5, "Layer", "bottom");   % nominal level
            end
            rows = rej(rej.region == r & rej.restriction == spec.restrictions(j), :);
            assert(isequal(rows.T, spec.T(:)));                  % rows in sample-size order
            for a = 1:3                                         % sample sizes at equally spaced positions 1:5
                hp(a) = plot(ax, 1:numel(spec.T), rows.(rate_vars(a)), st.levels(a).line, "Color", st.levels(a).color, ...
                    "Marker", st.levels(a).marker, "MarkerSize", 3.5, "MarkerFaceColor", st.levels(a).color, "LineWidth", st.line, ...
                    "Clipping", "off");                         % values at 0 or 1 sit on the frame
            end
            set(ax, "XTick", 1:numel(spec.T), "XTickLabel", string(spec.T));
            if rows.restriction_true(1)                         % restriction true for h0: zoom in on the nominal levels
                ylim(ax, [0 0.15]); yticks(ax, 0:0.05:0.15);
            else
                ylim(ax, [0 1.05]); yticks(ax, 0:0.25:1);       % headroom so rates of 1 clear the frame
            end
            xlim(ax, [0.7 numel(spec.T) + 0.3]); style_axes(ax, st); ax.XTickLabelRotation = 0;
            name = upper(extractBefore(spec.restrictions(j), 2)) + extractAfter(spec.restrictions(j), 1);
            if j == 1, name = sprintf("Region %d: [%.2f, %.2f]", r, spec.regions(r, :)) + newline + name; end
            title(ax, name, "FontSize", st.label, "FontWeight", "normal");
            if j == numel(spec.restrictions), xlabel(ax, "Sample size T", "FontSize", st.label); end
        end
    end
    lg = legend(hp, ["1%", "5%", "10%"], "Orientation", "horizontal", "FontSize", st.legend);   % one legend for all panels
    title(lg, "Nominal level", "FontWeight", "normal"); lg.Layout.Tile = "south";
    ylabel(t, "Rejection frequency", "FontName", st.font, "FontSize", st.label);
    save_figure(fig, fullfile(fig_dir, "mc_rejection_frequencies.pdf"));
end

function [fig, t] = paper_figure(width_cm, height_cm, rows, cols, spacing)
%% Invisible figure of the printed size (cm) with a tiled layout (1 x 1, compact spacing by default).
if nargin < 3, rows = 1; cols = 1; end
if nargin < 5, spacing = "compact"; end
fig = figure("Visible", "off", "Units", "centimeters", "Position", [2 2 width_cm height_cm], "Color", "w");
t = tiledlayout(fig, rows, cols, "TileSpacing", spacing, "Padding", "compact");
end

function style_axes(ax, st)
%% Common axes typography and frame: Times tick labels (tex interpreter keeps Times), grid, thin box.
set(ax, "FontName", st.font, "FontSize", st.tick, "TickLabelInterpreter", "tex", "LineWidth", 0.5, ...
    "Box", "on", "LabelFontSizeMultiplier", 1, "TitleFontSizeMultiplier", 1, "TitleFontWeight", "normal");
grid(ax, "on");
end

function save_figure(fig, file)
%% Vector PDF at the figure's own size (fonts embedded), then close.
exportgraphics(fig, file, "ContentType", "vector", "Padding", "figure");
close(fig);
end
