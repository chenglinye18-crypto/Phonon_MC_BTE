function LUTq = build_q_T_lookup_(spec, cfg)
% BUILD_Q_T_LOOKUP_  生成等温边界出射热流 q''(T) 与反解 T(q'') 的查表
% q''(T) = 1/4 * sum_{b,m} DOS * v_g * (ħω) * n_BE(T) * dw   [W/m^2]
%
% 必需: spec.DOS_w_b(B×Nw), spec.w_mid(B×Nw), spec.dw(1×Nw或B×Nw)
%       以及可用的群速 v_g(ω,b)。若无，则用 ensure_vg_table(spec) 预计算到 w_mid 上。
% 可选: cfg.T_min(默认1), cfg.T_max(默认2000), cfg.nT(默认2001)
%
% 返回:
%   LUTq.T    [nT×1]        温度网格 (K)
%   LUTq.q    [nT×1]        总出射热流 q''(T) (W/m^2)
%   LUTq.qb   [nT×B]        分支出射热流（调试用）
%   LUTq.inv  函数句柄      T = LUTq.inv(q_target)         （绝对式反解）
%   LUTq.inv_dev(Tref)      句柄 -> 返回函数 f(q_dev)=T    （偏差式反解）
%
  if nargin<2, cfg = struct(); end
  T_min = get_or_(cfg,'T_min',1);
  T_max = get_or_(cfg,'T_max',2000);
  nT    = get_or_(cfg,'nT',2001);

  % ---- 检查与取表 ----
  need = {'DOS_w_b','w_mid','dw'};
  for k=1:numel(need), if ~isfield(spec,need{k}), error('spec 缺少 %s',need{k}); end, end
  [B,Nw] = size(spec.DOS_w_b);
  if ~isequal(size(spec.w_mid),[B,Nw]), error('w_mid 尺寸需与 DOS_w_b 一致'); end

  if isvector(spec.dw)
      dw = reshape(spec.dw,1,[]);
      if numel(dw)~=Nw, error('dw 长度应为 Nw'); end
      dw = repmat(dw,B,1);
  else
      dw = spec.dw;
      if ~isequal(size(dw),[B,Nw]), error('dw 尺寸应为 1×Nw 或 B×Nw'); end
  end

  vg = ensure_vg_table(spec);     % B×Nw 的 |v_g|
  DOS = max(spec.DOS_w_b,0);
  w   = max(spec.w_mid,0);

  % ---- 常量与网格 ----
  hbar = 1.054571817e-34; kB = 1.380649e-23;
  T  = linspace(T_min,T_max,nT).';
  q  = zeros(nT,1);
  qb = zeros(nT,B);

  for it = 1:nT
      Ti = max(T(it),1e-9);
      x  = (hbar.*w)./(kB*Ti);                       % B×Nw
      nBE= 1 ./ max(exp(min(x,700)) - 1, realmin);   % 防溢出
      dq = 0.25 * DOS .* vg .* (hbar.*w) .* nBE .* dw; % B×Nw
      qb(it,:) = sum(dq,2).';                        % 每分支
      q(it)    = sum(qb(it,:));                      % 总
  end

  % ---- 反解：保证单调并构造句柄 ----
  [q_mono, T_mono] = make_monotone_(q, T);

  LUTq = struct();
  LUTq.T  = T;  LUTq.q  = q;  LUTq.qb = qb;
  LUTq.inv = @(qtar) invert_from_y_(qtar, q_mono, T_mono);      % 绝对式 q''(T)=qtar
  LUTq.inv_dev = @(Tref) @(qdev) invert_from_y_( qdev + q_at_(Tref,q_mono,T_mono), q_mono, T_mono );
end

% ======= 小工具（与您 U–T 版本保持一致风格）=======
function v = get_or_(s, name, dv)
  if isstruct(s) && isfield(s,name) && ~isempty(s.(name)), v = s.(name); else, v = dv; end
end

function [Yo, Xo] = make_monotone_(Y, X)
  Yo = Y(:);  for i=2:numel(Yo), if Yo(i)<Yo(i-1), Yo(i)=Yo(i-1); end, end
  Xo = X(:);
end

function Xout = invert_from_y_(Ytar, Ygrid, Xgrid)
  Ymin = Ygrid(1);  Ymax = Ygrid(end);
  Xout = interp1(Ygrid, Xgrid, min(max(Ytar, Ymin), Ymax), 'pchip','extrap');
end

function qref = q_at_(Tref, q_grid, T_grid)
  qref = interp1(T_grid, q_grid, Tref, 'pchip', 'extrap');
end

function vg_w_b = ensure_vg_table(spec)
  [B,Nw] = size(spec.w_mid); vg_w_b = zeros(B,Nw);
  for b=1:B
      for m=1:Nw
          w = spec.w_mid(b,m);
          [~, vabs] = local_q_vabs_from_w_table_(w, spec, b);
          vg_w_b(b,m) = vabs;
      end
  end
end

function [q, vabs] = local_q_vabs_from_w_table_(w, spec, b)
% 从材料表反解 q(w) 并取 |vg(q)|
  qv = spec.si.q(:);             % Nq×1
  wv = spec.si.omega_tab(b,:).'; % Nq×1
  gv = spec.si.vg_tab(b,:).';    % Nq×1

  [w_sorted, Is] = sort(max(wv,0), 'ascend');
  q_sorted = qv(Is); v_sorted = gv(Is);
  [ws, Iu] = unique(w_sorted, 'stable');
  qs = q_sorted(Iu); vs = v_sorted(Iu);

  w_cl = min(max(w, ws(1)), ws(end));
  q    = interp1(ws, qs, w_cl, 'pchip');
  v    = interp1(qs, vs, q,    'pchip');
  vabs = abs(v);
end
