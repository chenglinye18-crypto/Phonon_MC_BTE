function r_tau = precompute_relax_times_(state, Tcell, opts, spec)
% 返回 r_tau(Np×6)：[LA_BL, TA_N, TA_U, LTO, IMP, BOUND] 的“散射率” (s^-1)
% 依赖：表驱动 si.q, si.qmax, si.omega_tab(b,:), 以及粒子字段 p(i).{b,w,vabs,cell,v,q}

kB   = 1.380649e-23;  hbar = 1.054571817e-34;

Np = numel(state.p);
r_tau = zeros(Np,6);

% --- 常数,直接取的paper值 ---
B_L   = get_or(opts,'BL',   1.18e-24);       % s·K^-3  (LA)
B_TN  = get_or(opts,'BTN',  10.5e-13);       % K^-4    (TA Normal)
B_TU  = get_or(opts,'BTU',  2.89e-18);       % s       (TA Umklapp prefactor)
tau_LTO_ps = get_or(opts,'tau_LTO_ps', 3.5);% ps → 常数弛豫时间 (LO/TO)
A_imp = get_or(opts,'A_imp', 1.32e-45);     % s^3     (impurity杂质散射)

% 边界两种模型的参数（择一）：bulk 或 thin film
bulk_L = get_or(opts,'PB_bulk_L', 7.16e-3);      % m (给了就走 bulk)
bulk_F = get_or(opts,'PB_bulk_F', 0.68);    % -
Tsi    = get_or(opts,'PB_Tsi', []);         % m (给了就走薄膜)
Delta  = get_or(opts,'PB_Delta', 0);        % m (表中粗糙度Δ), 可为0→全漫反射 p≈0

% 运输法向（薄膜用来算 θ_B）；未提供则用各向同性平均 <cos^2>=1/3
n_hat = get_or(opts,'transport_n', [0;0;1]); n_hat = n_hat(:)/max(norm(n_hat),eps);

% --- TA 的 ω_cut = ω_TA(qmax/2)，用表插值 ---
bTA = find(strcmpi(spec.branches,'TA'),1);
if isempty(bTA), bTA = 2; end
omega_cut = interp1(spec.si.q(:), spec.si.omega_tab(bTA,:), 0.5*spec.qmax, 'linear', 'extrap');

% --- 主循环：逐粒子计算各散射率 ---
for i = 1:Np
    b  = state.p(i).b;
    w  = state.p(i).w;            % rad/s
    vg = state.p(i).vabs;         % m/s
    T  = max(Tcell(state.p(i).cell), 1e-6);

    % 1) LA BL
    if is_LA(spec,b)
        r_tau(i,1) = B_L * (w.^2) * (T.^3);
    end

    % 2) TA N
    if is_TA(spec,b)
        r_tau(i,2) = B_TN * w * (T.^4);
    end

    % 3) TA U with cutoff
    if is_TA(spec,b)
        if w > omega_cut
            x = hbar*w/(kB*T);
            r_tau(i,3) = B_TU * (w.^2) / max(sinh(x), 1e-12);
        else
            r_tau(i,3) = 0;
        end
    end

    % 4) L/TO 常数弛豫
    if is_LO(spec,b) || is_TO(spec,b)
        r_tau(i,4) = 1 / (tau_LTO_ps * 1e-12);   % s^-1
    end

    % 5) 杂质  A ω^4
    r_tau(i,5) = A_imp * (w.^4);

    % 6) 边界：bulk 或 thin-film（二者择一，若都给以 thin-film 优先）
    if ~isempty(Tsi)           % 薄膜（Ref.30）
        % cosθ_B 取瞬时方向或各向同性平均
        v = state.p(i).v(:);
        if all(isfinite(v)) && norm(v)>0
            cosB = abs(dot(v, n_hat)) / (norm(v)+eps);
        else
            cosB = sqrt(1/3);  % <cos^2>^0.5
        end
        qbar = max(state.p(i).q, 0);     % 波矢模
        p_spec = exp( -4*(qbar*Delta)^2 * (cosB^2) );
        Ffilm  = (1 - p_spec) / (1 + p_spec);     % 表中因子
        r_tau(i,6) = vg / max(Tsi,1e-12) * Ffilm;
    elseif ~isempty(bulk_L)    % 体块（Ref.29）
        r_tau(i,6) = vg / max(bulk_L*bulk_F, 1e-12);
    else
        r_tau(i,6) = 0;
    end
end
end

% --- 小工具与分支识别 ---
function tf = is_LA(spec,b), tf = branch_is(spec,b,'LA'); end
function tf = is_TA(spec,b), tf = branch_is(spec,b,'TA'); end
function tf = is_LO(spec,b), tf = branch_is(spec,b,'LO'); end
function tf = is_TO(spec,b), tf = branch_is(spec,b,'TO'); end
function tf = branch_is(spec,b,tag)
    nm = upper(strrep(spec.branches(b),' ',''));
    tf = contains(nm, upper(tag));  % 兼容比如 'TA1','TA2'
end
function v = get_or(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
