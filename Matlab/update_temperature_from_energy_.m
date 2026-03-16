function [Tnew, aux] = update_temperature_from_energy_(state, mesh, spec, opts, LUT)
% UPDATE_TEMPERATURE_FROM_ENERGY_
% 将粒子能量(有符号)聚合到cell → 转能量密度 → 查表反解温度
% 口径：
%   - absolute:   Uabs = (∑E_i)/V
%   - deviational:Uabs = Udev + Uref,  其中 Udev = (∑E_i)/V，Uref = Ueq(Tref)
%
% 依赖：
%   - LUT = build_E_T_lookup_(spec, cfg)，其中 LUT.T, LUT.Ueq（可选 LUT.Uref）
%   - mesh 提供规则网格边界或 boxes，用于 cell 体积计算
%
% 打印：Np、总能量、温度统计（并标注 abs/dev）

  if nargin < 4 || isempty(opts), opts = struct(); end
  if ~isfield(opts,'E_eff'), opts.E_eff = 1e-18; end
  if ~isfield(opts,'mode'),  opts.mode  = 'absolute'; end

  % ---- LUT：若未提供则构建（deviational 时可顺便缓存 Uref）----
  if nargin < 5 || isempty(LUT)
      cfg = struct('T_min',1,'T_max',2000,'nT',2001);
      if strcmpi(opts.mode,'deviational') && isfield(opts,'Tref') && isscalar(opts.Tref)
          cfg.Tref = opts.Tref;  % 便于缓存 Uref
      end
      LUT = build_E_T_lookup_(spec, cfg);
  end

  % ---- cell 体积与 Nc ----
  Vc = cell_volumes_(mesh);
  Nc = numel(Vc);
  if Nc==0
      Tnew = []; aux = struct(); return;
  end

  % ---- 合法粒子索引（1..Nc）----
  if isempty(state.p)
      cid = int32([]); valid = false(0,1);
  else
      cid_all = int32([state.p.cell].');
      valid   = (cid_all >= 1) & (cid_all <= Nc);
      cid     = cid_all(valid);
  end

  % ---- 每 cell 能量（有符号）----
  if isempty(cid)
      Ecell = zeros(Nc,1);
  elseif isfield(state.p,'E')
      Epart_all = [state.p.E].';            % 有符号
      Epart     = Epart_all(valid);
      Ecell     = accumarray(double(cid), Epart, [Nc 1], @sum, 0);   % J
  else
      % 兜底（极少用到）：按计数×E_eff，并尝试用 sgn 字段恢复符号
      cnt = accumarray(double(cid), 1, [Nc 1], @sum, 0);
      if ~isempty(state.p) && isfield(state.p,'sgn')
          sgn_all = [state.p.sgn].';
          sgn_acc = accumarray(double(cid), sgn_all(valid), [Nc 1], @sum, 0);
          sgn_avg = ones(Nc,1); msk = cnt>0; sgn_avg(msk) = sgn_acc(msk)./max(cnt(msk),1);
          sgn_avg(sgn_avg==0) = 1;
      else
          sgn_avg = ones(Nc,1);
      end
      Ecell = (opts.E_eff .* cnt) .* sgn_avg;
  end

  % ---- 绝对/偏差两种口径：构造 Uabs ----
  Ulocal = zeros(Nc,1);                     % J/m^3
  mV = Vc > 0;
  Ulocal(mV) = Ecell(mV) ./ Vc(mV);         % absolute: 就是 Uabs；deviational: 这是 Udev

  if strcmpi(opts.mode,'deviational')
      Tref_cell = local_reference_temperature_field_(state, opts, Nc);
      if isfield(LUT,'Uref') && isfinite(LUT.Uref) && isscalar(Tref_cell)
          Uref = LUT.Uref;
      else
          Tref_cl = min(max(Tref_cell, LUT.T(1)), LUT.T(end));
          Uref = pchip(LUT.T, LUT.U, Tref_cl);
      end
      Uabs = Ulocal + Uref;                 % J/m^3
  else
      Uabs = Ulocal;                         % absolute
  end

  % ---- 反解 Uabs → T（范围夹取+单调 pchip）----
  Umin = LUT.U(1); Umax = LUT.U(end);
  Ucl  = min(max(Uabs, Umin), Umax);
  Tnew = pchip(LUT.U, LUT.T, Ucl);

  % ---- 打印 ----
  mode_str = 'abs'; if strcmpi(opts.mode,'deviational'), mode_str='dev'; end
  total_particles = numel(cid);
  total_energy    = sum(Ecell);             % J（偏差法为净偏差能量之和）
  if isempty(Tnew), Tmin=NaN; Tmean=NaN; Tmax=NaN;
  else, Tmin=min(Tnew); Tmean=mean(Tnew); Tmax=max(Tnew);
  end
  fprintf(1,'[temp-update:%s] Np=%d | E_total=%.6e J | T[min,mean,max]=[%.2f, %.2f, %.2f] K\n', ...
      mode_str, total_particles, total_energy, Tmin, Tmean, Tmax);

  % ---- 诊断 ----
  aux = struct();
  aux.Ecell    = Ecell;     % J
  aux.Vcell    = Vc;        % m^3
  aux.Ulocal   = Ulocal;    % abs: Uabs；dev: Udev
  aux.Uabs     = Uabs;      % 送去查表的绝对能量密度
  aux.clip_low  = mean(Uabs < Umin);
  aux.clip_high = mean(Uabs > Umax);
  aux.LUT       = LUT;
end

function Tref_cell = local_reference_temperature_field_(state, opts, Nc)
  if isstruct(state) && isfield(state, 'info') && isstruct(state.info) && ...
          isfield(state.info, 'Tref_cell') && numel(state.info.Tref_cell) == Nc
      Tref_cell = state.info.Tref_cell(:);
      return;
  end

  if isfield(opts, 'Tref_cell') && numel(opts.Tref_cell) == Nc
      Tref_cell = opts.Tref_cell(:);
      return;
  end

  if isfield(opts, 'Tref') && isfinite(opts.Tref)
      Tref_cell = opts.Tref * ones(Nc, 1);
      return;
  end

  error('update_temperature_from_energy_: deviational 模式需提供 Tref_cell 或 opts.Tref.');
end

% ====================== 构建 E(T) 查表（能量密度） ======================
function LUT = build_E_T_lookup_(spec, cfg)
% 返回：
%   LUT.T (nT×1), LUT.U (nT×1)，可选 LUT.Uref（若 cfg.Tref 提供）
% 需要 spec:
%   spec.w_mid (B×Nw)     频率（rad/s）
%   spec.dw    (1×Nw)     频带宽
%   spec.DOS_w_b (B×Nw) 或 spec.N_w(B×Nw) —— DOS≈N_w/Δω 近似

  arguments
    spec struct
    cfg.T_min (1,1) double = 1
    cfg.T_max (1,1) double = 2000
    cfg.nT    (1,1) double = 2001
    cfg.Tref  (1,1) double = NaN
  end

  kB   = 1.380649e-23;
  hbar = 1.054571817e-34;

  Tvec = linspace(cfg.T_min, cfg.T_max, cfg.nT).';
  [B, Nw] = size(spec.w_mid);

  % Δω
  if isfield(spec,'dw') && numel(spec.dw)==Nw
      dw = reshape(spec.dw, 1, Nw);
  else
      error('build_E_T_lookup_: spec.dw (1×Nw) is required.');
  end

  % DOS
  if isfield(spec,'DOS_w_b') && ~isempty(spec.DOS_w_b)
      DOS = max(spec.DOS_w_b, 0);           % B×Nw
  elseif isfield(spec,'N_w') && ~isempty(spec.N_w)
      DOS = max(spec.N_w,0) ./ repmat(dw, B,1);
  else
      error('build_E_T_lookup_: need spec.DOS_w_b or spec.N_w.');
  end

  % 向量化计算 Ueq(T) = Σ_bm DOS(b,m) * ħω(b,m) * n_BE(ω,T) * Δω
  Wb = hbar .* spec.w_mid;                   % B×Nw
  Ueq = zeros(numel(Tvec),1);
  for i = 1:numel(Tvec)
      T   = max(Tvec(i), 1e-12);
      x   = Wb ./ (kB*T);                    % B×Nw
      nBE = 1 ./ max(exp(min(x,700)) - 1, realmin);
      Ueq(i) = sum( (DOS .* Wb .* nBE) .* repmat(dw,B,1), 'all' );
  end

  % 保证单调（理论上单调递增），若有数值抖动则台阶修正
  [Ueq, Tvec] = ensure_monotonic_(Ueq, Tvec);

  LUT = struct('T', Tvec, 'U', Ueq, 'Ueq', Ueq);

  % 可选：缓存 Uref（deviational 用）
  if isfield(cfg,'Tref') && isfinite(cfg.Tref)
      LUT.Uref = pchip(LUT.T, LUT.U, max(cfg.Tref, LUT.T(1)));
  end
end

% ====================== 反解：能量密度 → 温度（备用接口） ======================
function T = invert_T_from_U_(U, LUT)
% 仅作为备用；主流程已直接用 pchip(LUT.Ueq→LUT.T)
  Umin = LUT.U(1); Umax = LUT.U(end);
  Ucl  = min(max(U, Umin), Umax);
  T    = pchip(LUT.U, LUT.T, Ucl);
end

% ====================== 辅助：保证单调 ======================
function [y2, x2] = ensure_monotonic_(y, x)
  y2 = y(:); x2 = x(:);
  for i = 2:numel(y2)
      if y2(i) < y2(i-1)
          y2(i) = y2(i-1);
      end
  end
end

% ====================== cell 体积 ======================
function Vc = cell_volumes_(mesh)
  if all(isfield(mesh, {'Nx','Ny','Nz','x_edges','y_edges','z_edges'}))
      Nx=mesh.Nx; Ny=mesh.Ny; Nz=mesh.Nz;
      dx = diff(mesh.x_edges(:)); dy = diff(mesh.y_edges(:)); dz = diff(mesh.z_edges(:));
      Vc3 = reshape(dx,[],1) .* reshape(dy,1,[]) .* reshape(dz,1,1,[]);
      Vc  = Vc3(:);       % Nc×1
  elseif isfield(mesh,'boxes') && ~isempty(mesh.boxes)
      bx  = mesh.boxes;
      Vc  = (bx(:,2)-bx(:,1)) .* (bx(:,4)-bx(:,3)) .* (bx(:,6)-bx(:,5));
  elseif isfield(mesh,'cell_vol') && ~isempty(mesh.cell_vol)
      Vc = mesh.cell_vol(:);
  else
      error('cell_volumes_: missing mesh edges/boxes/cell_vol.');
  end
end
