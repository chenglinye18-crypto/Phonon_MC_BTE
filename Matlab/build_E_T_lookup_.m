function LUT = build_E_T_lookup_(spec, cfg)
% BUILD_E_T_LOOKUP_  生成平衡能量密度 U(T) 与反解 T(U) 的查表
% U(T) = sum_p ∫ D(ω,p) * ħω * f_BE(ω,T) dω  （单位: J/m^3）
%
% 输入:
%   spec.DOS_w_b  [B×Nw]   每个分支的 DOS(ω)
%   spec.w_mid    [B×Nw]   每个 bin 的中心频率 (rad/s)
%   spec.dw       [1×Nw] 或 [B×Nw]  频率步长
% 可选:
%   cfg.T_min (默认 1 K), cfg.T_max (默认 2000 K), cfg.nT (默认 2001)
%
% 返回:
%   LUT.T   [nT×1]        温度网格 (K)
%   LUT.U   [nT×1]        平衡能量密度 (J/m^3)
%   LUT.inv 函数句柄      T = LUT.inv(U_target)  （向量化，夹取）
%   LUT.Ub  [nT×B]        分支分解的能量密度（调试/诊断用）

  if nargin < 2, cfg = struct(); end
  T_min = get_or_(cfg, 'T_min', 1);
  T_max = get_or_(cfg, 'T_max', 2000);
  nT    = get_or_(cfg, 'nT',    2001);

  % 安全检查与取网格
  if ~isfield(spec,'DOS_w_b') || ~isfield(spec,'w_mid') || ~isfield(spec,'dw')
      error('build_E_T_lookup_: spec 需含 DOS_w_b, w_mid, dw');
  end
  [B, Nw] = size(spec.DOS_w_b);
  if ~isequal(size(spec.w_mid), [B, Nw])
      error('spec.w_mid 尺寸需与 DOS_w_b 一致 (B×Nw)');
  end
  if isvector(spec.dw)
      dw = reshape(spec.dw, 1, []);
      if numel(dw) ~= Nw, error('spec.dw 长度应为 Nw'); end
      dw = repmat(dw, B, 1);
  else
      dw = spec.dw;
      if ~isequal(size(dw), [B, Nw]), error('spec.dw 尺寸应为 1×Nw 或 B×Nw'); end
  end

  % 常数
  hbar = 1.054571817e-34;     % J·s
  kB   = 1.380649e-23;        % J/K

  T = linspace(T_min, T_max, nT).';
  U  = zeros(nT,1);
  Ub = zeros(nT,B);

  % 逐温度积分（向量化到分支/频率）
  DOS = max(spec.DOS_w_b, 0);
  w   = max(spec.w_mid,   0);

  for it = 1:nT
      Ti = max(T(it), 1e-9);
      x  = (hbar .* w) ./ (kB * Ti);         % B×Nw
      % 防溢出：exp(>~700) 溢出
      nBE = 1 ./ max(exp(min(x,700)) - 1, realmin);
      dE  = DOS .* (hbar .* w) .* nBE .* dw; % B×Nw
      Ub(it,:) = sum(dE, 2).';               % 每个分支的能量密度
      U(it)    = sum(Ub(it,:));              % 总能量密度
  end

  % 反解：单调插值（若有数值噪声，做一点单调修正）
  % 确保严格单调
  [U_mono, T_mono] = make_monotone_(U, T);

  LUT = struct();
  LUT.T  = T;
  LUT.U  = U;
  LUT.Ub = Ub;
  LUT.inv = @(Utarget) invert_T_from_energy_density_(Utarget, U_mono, T_mono);
end

% ======= 工具们 =======
function v = get_or_(s, name, dv)
  if isstruct(s) && isfield(s,name) && ~isempty(s.(name)), v = s.(name); else, v = dv; end
end

function [Uo, To] = make_monotone_(U, T)
  % 若 U(T) 有微小非单调（数值误差），用累积最大值修正
  Uo = U(:);
  for i = 2:numel(Uo)
      if Uo(i) < Uo(i-1), Uo(i) = Uo(i-1); end
  end
  To = T(:);
end

function Tout = invert_T_from_energy_density_(Utarget, Ugrid, Tgrid)
  % 向量化反解：给出能量密度 Utarget（可为标量/向量），返回 T
  Umin = Ugrid(1);  Umax = Ugrid(end);
  Tout = interp1(Ugrid, Tgrid, min(max(Utarget, Umin), Umax), 'pchip', 'extrap');
end
