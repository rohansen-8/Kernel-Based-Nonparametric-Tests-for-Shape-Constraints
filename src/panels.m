function varargout = panels(op, varargin)
%% Data module: E1 beta-decile panel (load; build with the construction rule), E2 characteristic
%   portfolios, and the E3/E4 Treasury-bill panel.
%   Usage: panels("<operation>", arguments...). Operations:
%     "beta"             load_beta_panel
%     "beta_rebuild"     build_beta_panel
%     "pt"               build_pt_portfolios
%     "treasury"         build_treasury_panel
%     "treasury_month"   treasury_month_panel
switch op
    case "beta", f = @load_beta_panel;
    case "beta_rebuild", f = @build_beta_panel;
    case "pt", f = @build_pt_portfolios;
    case "treasury", f = @build_treasury_panel;
    case "treasury_month", f = @treasury_month_panel;
    otherwise, error("panels: unknown operation ""%s""", op);
end
[varargout{1:nargout}] = f(varargin{:});
end

function panel = load_beta_panel(root)
%% Load the E1 beta-decile panel (one row per formation month, one column per decile).
%   x(t,j): value-weighted portfolio beta_60m of decile j at formation month t
%   Y(t,j): value-weighted excess return of decile j over the following month

S = load(fullfile(root, "data", "processed", "beta_decile_panel_1975_2025.mat"), "panel");
panel.formation_month = S.panel.formation_month(:);
panel.return_month = S.panel.return_month(:);
panel.x = S.panel.Z;
panel.Y = S.panel.portfolio_return;
panel.number_firms = S.panel.number_firms;
end

function panel = build_beta_panel(jkp_dir, months)
%% Rebuild the E1 beta-decile panel for the given formation months (month-end datetimes) from the annual JKP files,
%   with the construction rule of the panel used in the paper (README, "Data vintage"): keep stocks with finite id, beta_60m,
%   me > 0 and ret_exc_lead1m; sort by (beta_60m, id); decile j = ceil(10 i / n); value-weighted decile beta Z and
%   next-month excess return. Returns the panel only; scripts/build_data.m saves it as the E1 input
%   data/processed/beta_decile_panel_1975_2025.mat when that file is absent. Files without an `id` column (the later
%   vintage of the 2024-2025 files) are rejected, because that vintage does not reproduce the paper's panel.
months = dateshift(months(:), "end", "month");
vars = ["id", "eom", "me", "ret_exc_lead1m", "beta_60m"];
n = numel(months);
[Z, R] = deal(NaN(n, 10)); N = zeros(n, 10);
for yy = unique(year(months))'
    file = fullfile(jkp_dir, sprintf("jkp_global_factor_usa_%d.csv", yy));
    opts = detectImportOptions(file);
    if ~all(ismember(vars, string(opts.VariableNames)))
        error("%s lacks %s: it is not the JKP vintage of the E1 panel.", file, strjoin(setdiff(vars, string(opts.VariableNames)), ", "));
    end
    opts.SelectedVariableNames = cellstr(vars);
    D = readtable(file, opts);
    if ~isdatetime(D.eom), D.eom = datetime(D.eom); end
    for t = find(year(months) == yy)'
        M = D(D.eom == months(t), :);
        M = M(isfinite(M.id) & isfinite(M.beta_60m) & isfinite(M.me) & M.me > 0 & isfinite(M.ret_exc_lead1m), :);
        M = sortrows(M, ["beta_60m", "id"]);
        decile = ceil((1:height(M))' * 10 / height(M));
        for j = 1:10
            w = M.me(decile == j) / sum(M.me(decile == j));
            N(t, j) = sum(decile == j);
            Z(t, j) = sum(w .* M.beta_60m(decile == j));
            R(t, j) = sum(w .* M.ret_exc_lead1m(decile == j));
        end
    end
end
panel = struct("formation_month", months, "return_month", dateshift(months + calmonths(1), "end", "month"), ...
    "Z", Z, "portfolio_return", R, "Y", mean(R, 2), "number_firms", N);
end

function pt = build_pt_portfolios(root, years)
%% Build monthly value-weighted decile portfolios on the eight Patton-Timmermann (2010, Table 4) characteristics.
%   Each JKP year file is read once. In every formation month t, stocks with positive market equity, a finite
%   next-month excess return and a finite characteristic are sorted into ten equal-count deciles (ties broken by
%   permno). As in the Ken French portfolios used by PT, nonpositive values are excluded for the four ratio
%   characteristics (book-to-market, cash flow-, earnings-to-price and dividend yield).
%   x(t,j,k): value-weighted characteristic of decile j for characteristic k
%   Y(t,j,k): value-weighted excess return of decile j over month t+1
%   n(t,j,k): number of stocks in decile j
pt.names = ["Size", "Book-to-market", "Cash flow-to-price", "Earnings-to-price", "Dividend yield", ...
    "Momentum", "Short-term reversal", "Long-term reversal"];
pt.columns = ["me", "be_me", "ocf_me", "ni_me", "div12m_me", "ret_12_1", "ret_1_0", "ret_60_12"];
pt.direction = [-1 1 1 1 1 1 -1 -1];          % PT hypothesized direction (+1 increasing, -1 decreasing)
pt.positive_only = [false true true true true false false false];
K = numel(pt.columns);
J = 10;

months = []; x = []; Y = []; n = [];
for year = years(:)'
    file = fullfile(root, "data", "raw", "jkp", "jkp_global_factor_usa_" + year + ".csv");
    vars = ["eom", "permno", "ret_exc_lead1m", pt.columns];
    opts = detectImportOptions(file, "VariableNamingRule", "preserve");
    opts.SelectedVariableNames = vars;
    opts = setvartype(opts, vars(2:end), "double");
    d = readtable(file, opts);
    d.eom = dateshift(d.eom, "end", "month");     % 2024-25 files use trading-day month ends
    year_months = unique(d.eom);
    x_y = nan(numel(year_months), J, K); Y_y = x_y; n_y = zeros(numel(year_months), J, K);
    for m = 1:numel(year_months)
        dm = d(d.eom == year_months(m), :);
        base = dm.me > 0 & isfinite(dm.ret_exc_lead1m);
        for k = 1:K
            c = dm.(pt.columns(k));
            valid = base & isfinite(c) & (~pt.positive_only(k) | c > 0);
            if nnz(valid) < J, continue, end
            [~, order] = sortrows([c(valid), dm.permno(valid)]);
            rank(order) = 1:nnz(valid); %#ok<AGROW>
            decile = ceil(J * rank(1:nnz(valid))' / nnz(valid));
            me = dm.me(valid);
            x_y(m, :, k) = accumarray(decile, me .* c(valid), [J 1])' ./ accumarray(decile, me, [J 1])';
            Y_y(m, :, k) = accumarray(decile, me .* dm.ret_exc_lead1m(valid), [J 1])' ./ accumarray(decile, me, [J 1])';
            n_y(m, :, k) = accumarray(decile, 1, [J 1])';
            clear rank
        end
    end
    months = [months; year_months]; x = [x; x_y]; Y = [Y; Y_y]; n = [n; n_y]; %#ok<AGROW>
    fprintf("%d: %d months, %d rows\n", year, numel(year_months), height(d));
end
pt.formation_month = months;
pt.return_month = dateshift(months, "end", "month", "next");
pt.x = x; pt.Y = Y; pt.n = n;
end

function tp = build_treasury_panel(root)
%% Build the monthly Treasury-bill panel used by E3 and E4 from the CRSP monthly Treasury and risk-free files.
%   One row per bill (ITYPE 4) and formation month t with a return in month t+1:
%   maturity: remaining maturity at formation, months (days * 12 / 365.25)
%   Y:        bill return over t -> t+1 (TMRETNUA at t+1) minus the one-month risk-free return, where the
%             risk-free return compounds the CRSP 1-month yield at t (TTERMTYPE 30, TMYTM) daily over the holding days
%   r:        formation-date CRSP 1-month yield, annualized percent (short-rate state for E4)
%   amount:   amount outstanding at formation (TMTOTOUT), used for within-month weights
%   Bills must still be outstanding at t+1 and have a positive amount and finite returns. No maturity bounds.
raw = fullfile(root, "data", "raw", "crsp");
opts = detectImportOptions(fullfile(raw, "crsp_treasury_monthly_1964_2025.csv"));
opts.SelectedVariableNames = ["KYTREASNO", "TMATDT", "ITYPE", "MCALDT", "TMRETNUA", "TMTOTOUT"];
B = readtable(fullfile(raw, "crsp_treasury_monthly_1964_2025.csv"), opts);
B = B(B.ITYPE == 4, :);
R = readtable(fullfile(raw, "crsp_riskfree_monthly_1964_2025.csv"));
R = sortrows(R(R.TTERMTYPE == 30, :), "MCALDT");

rf = table(R.MCALDT(1:end - 1), R.MCALDT(2:end), R.TMYTM(1:end - 1), ...
    'VariableNames', ["formation_date", "return_date", "r"]);
rf.rf_return = (1 + rf.r / 36500) .^ days(rf.return_date - rf.formation_date) - 1;

F = table(B.KYTREASNO, B.MCALDT, B.TMATDT, B.TMTOTOUT, 'VariableNames', ["id", "formation_date", "maturity_date", "amount"]);
S = table(B.KYTREASNO, B.MCALDT, B.TMRETNUA, 'VariableNames', ["id", "return_date", "bill_return"]);
tp = innerjoin(innerjoin(F, rf, "Keys", "formation_date"), S, "Keys", ["id", "return_date"]);
tp.maturity = days(tp.maturity_date - tp.formation_date) * 12 / 365.25;
tp.Y = tp.bill_return - tp.rf_return;
keep = tp.maturity_date > tp.return_date & tp.amount > 0 & isfinite(tp.amount) & isfinite(tp.Y);
tp = sortrows(tp(keep, ["formation_date", "return_date", "id", "maturity", "Y", "r", "amount", "bill_return", "rf_return"]), ...
    ["formation_date", "maturity"]);
end

function [X, Y, A, months, R] = treasury_month_panel(d)
%% Arrange Treasury rows (sorted by formation_date) into month x bill panels of maturity, excess return, amount
%  and the formation-date short rate.
[g, months] = findgroups(d.formation_date);
first = accumarray(g, (1:height(d))', [], @min);
pos = (1:height(d))' - first(g) + 1;
idx = sub2ind([numel(months), max(pos)], g, pos);
[X, Y, A, R] = deal(nan(numel(months), max(pos)));
X(idx) = d.maturity; Y(idx) = d.Y; A(idx) = d.amount; R(idx) = d.r;
end
