function [dt, info] = step_pick_dt_(mesh, spec, opts, state, rates_mat)
% 选 Δt：CFL 与“单步散射概率”两个约束，兼容 Np×6 或 Nc×6 的速率矩阵

% ---- 读参数/默认 ----
dt_min   = get_or(opts,'dt_min',1e-15);
dt_max   = get_or(opts,'dt_max',1e-10);
cfl_safe = get_or(opts,'dt_safety_cfl',0.5);
p_target = get_or(opts,'p_target',0.01);
mode     = lower(get_or(opts,'dt_prob_mode','max'));   % 'max'|'avg'|'pctl'
pctl     = get_or(opts,'dt_prob_pctl',95);

Nc = mesh.Nc;
Np = numel(state.p);

% ---- CFL 约束 ----
% vg_max：优先 spec.vg_max，否则从粒子表估计
if isfield(spec,'vg_max') && ~isempty(spec.vg_max)
    vg_max = max(1e-9, spec.vg_max);
elseif Np>0
    vabs = arrayfun(@(pp) pp.vabs, state.p);
    vg_max = max(1e-9, max(vabs));
else
    vg_max = 6000; % 兜底
end

% hmin：优先 mesh.hmin；否则由 boxes 估算
if isfield(mesh,'hmin') && ~isempty(mesh.hmin)
    hmin = mesh.hmin;
elseif isfield(mesh,'boxes') && ~isempty(mesh.boxes)
    dx = mesh.boxes(:,2)-mesh.boxes(:,1);
    dy = mesh.boxes(:,4)-mesh.boxes(:,3);
    dz = mesh.boxes(:,6)-mesh.boxes(:,5);
    hmin = max(1e-12, min([dx(:);dy(:);dz(:)]));
else
    hmin = 1.0; % 兜底
end
dt_cfl = cfl_safe * hmin / vg_max;

% ---- 概率约束 ----
% 单步总散射概率 P = 1 - exp(-dt * r_tot) ≤ p_target
% ⇒ dt ≤ -log(1-p_target) / r_stat
r_stat = 0; dt_prob = inf;
if nargin >= 5 && ~isempty(rates_mat)
    [nrow,~] = size(rates_mat);
    r_tot_vec = sum(rates_mat, 2);      % Np×1

    r_pos = r_tot_vec(r_tot_vec > 0);
    if ~isempty(r_pos)
        switch mode
            case 'max'
                r_stat = max(r_pos);
            case 'avg'
                r_stat = mean(r_pos);
            case 'pctl'
                r_stat = prctile(r_pos, pctl);
            otherwise
                r_stat = max(r_pos);
        end
        dt_prob = -log(max(1 - p_target, 1e-12)) / max(r_stat, 1e-30);
    end
end

% ---- 取最小并夹在范围内 ----
dt_raw = min(dt_cfl, dt_prob);
dt     = min(max(dt_raw, dt_min), dt_max);

% ---- 诊断 ----
info = struct();
info.dt_cfl   = dt_cfl;
info.dt_prob  = dt_prob;
info.dt_raw   = dt_raw;
info.dt       = dt;
info.vg_max   = vg_max;
info.hmin     = hmin;
info.p_target = p_target;
info.mode     = mode;
info.r_stat   = r_stat;
end

function v = get_or(s, name, default_v)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default_v;
    end
end
