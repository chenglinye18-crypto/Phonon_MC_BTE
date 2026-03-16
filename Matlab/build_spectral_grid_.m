function spec = build_spectral_grid_(mat, opts)
% build_spectral_grid_  基于离散色散表构建蒙卡用的谱离散与抽样权重（全查表）
%
% INPUT:
%   mat   表驱动材料结构（见 mat_silicon_100 表驱动版）
%         需要字段： .q, .qmax, .branch_names, .degeneracy, .omega_tab(B×Mtab),
%                   .vg_tab(B×Mtab), .omega(b,q), .vg(b,q)
%   opts.T0   线化温度 [K]   （必须）
%   opts.n_q  q-网格数（默认 64，用于构建面/体权重与 Cv^{bin}）
%   opts.n_w  频率网格数（默认 1000，用于 DOS(ω) 等）
%   opts.weight_by_Cv_for_Q (体源按 Cv 分配, 默认 true)
%
% OUTPUT (spec):
%   —— q-网格（用于抽样与 Cv^{bin}）
%   .si, .T0, .q_edges, .q_mid, .dq, .qmax
%   .branches, .deg, .B, .M
%   .omega(B×M), .vg(B×M), .Cv(B×M), .Cv_tot
%   .face_weight(B×M), .vol_weight(B×M), .cdf_face(1×B*M), .cdf_vol(1×B*M)
%   .bm2ind, .ind2bm, .vg_max
%
%   —— 频率网格（用于 DOS 与按 ω 的统计）
%   .w_edges(1×Nw+1), .w_mid(B×Nw is just repetition per branch), .dw
%   .DOS_w(1×Nw), .DOS_w_b(B×Nw)
%   .N_w(B×Nw), .N_w_tot
%   .vg_w(B×Nw)   % 与 DOS 的权重一致的“有效”群速：vg_eff = (∑ q^2)/(∑ q^2/|vg|)

  % ---------- 常量/默认 ----------
  kB   = 1.380649e-23;
  hbar = 1.054571817e-34;

  if ~isfield(opts,'T0'), error('opts.T0 必须提供'); end
  if ~isfield(opts,'n_q'), opts.n_q = 5000; end
  if ~isfield(opts,'n_w'), opts.n_w = 1000; end
  if ~isfield(opts,'weight_by_Cv_for_Q'), opts.weight_by_Cv_for_Q = true; end

  T0 = opts.T0; Mq = opts.n_q; Nw = opts.n_w;

  % ---------- q-网格（等分 0..qmax；用来做 Cv^{bin} 与抽样权重） ----------
  q_edges = linspace(0, mat.qmax, Mq+1);
  q_mid   = 0.5*(q_edges(1:end-1) + q_edges(2:end));
  dq      = diff(q_edges);

  branches = mat.branch_names;
  deg      = mat.degeneracy;
  B        = numel(branches);
  M        = Mq;

  omega = zeros(B,M);
  vg    = zeros(B,M);
  Cv    = zeros(B,M);

  % —— 用 3D 各向同性 q-壳体权重 g(q)Δq=(deg/2π^2)q^2Δq 构造 Cv^{bin}(T0)（全查表）——
  for b = 1:B
    w_b  = mat.omega(b, q_mid);                     % 查表 ω(q)
    w_b  = max(w_b, 0);
    vg_b = mat.vg(b, q_mid);                        % 查表 v_g(q)

    x     = hbar.*w_b./(kB*T0);
    ex    = exp(x);                       % 数值稳健
    nbar  = 1./max(ex-1, realmin);
    dndT  = (hbar.*w_b./(kB*T0.^2)) .* nbar .* (nbar+1);

    dos_q = (deg(b)/(2*pi^2)) * (q_mid.^2) .* dq;   % q-壳体态数
    omega(b,:) = w_b;
    vg(b,:)    = vg_b;
    Cv(b,:)    = (hbar.*w_b) .* dndT .* dos_q;      % [J/(m^3·K)]
  end

  Cv_tot = sum(Cv(:));
  vg_max = max(abs(vg(:)));

  % —— 面/体源抽样（q-网格）——
  face_weight = abs(vg) .* Cv;                  % 面源 ∝ |v_g| C_v
  if opts.weight_by_Cv_for_Q
      vol_weight = Cv;                          % 体源 ∝ C_v
  else
      vol_weight = ones(size(Cv));              % 或者均匀
  end
  wf = reshape(face_weight, 1, []);
  wv = reshape(vol_weight,  1, []);
  cdf_face = cumsum(wf); if cdf_face(end)>0, cdf_face = cdf_face / cdf_face(end); else, cdf_face(:)=0; end
  cdf_vol  = cumsum(wv); if cdf_vol(end )>0, cdf_vol  = cdf_vol  / cdf_vol(end ); else, cdf_vol(:)=0; end

  % ---------- 频率网格（统一覆盖所有分支的 ω 范围） ----------
  % 直接用材料表的最小/最大 ω（更精细）
  wtab_all = max(mat.omega_tab, 0);
  wmin = max(0, min(wtab_all(:)));
  wmax = max(wtab_all(:));
  if ~(isfinite(wmin) && isfinite(wmax)) || wmax <= wmin
      % 极端兜底
      wmin = 0; wmax = 1.0;
  end
  w_edges   = linspace(wmin, wmax, Nw+1);
  w_mid_1x  = 0.5*(w_edges(1:end-1) + w_edges(2:end));
  dw        = diff(w_edges);
  % 展开成 B×Nw（便于与其它量同维）
  w_mid = repmat(w_mid_1x, B, 1);

  % ---------- 分支态密度 DOS_w_b 与 vg_w（全查表：分段单调反插值） ----------
  DOS_w_b = zeros(B, Nw);
  vg_w    = zeros(B, Nw);

  for b = 1:B
      qtab  = mat.q(:).';
      wtab  = max(mat.omega_tab(b,:), 0);
      vgtab = abs(mat.vg_tab(b,:));

      % —— 将 w(q) 按单调段拆分，逐段反插值 —— 
      d = diff(wtab);
      % 段断点：导数变号或出现平坦点
      brk = 1;
      for i = 2:numel(d)
          if d(i-1)*d(i) < 0 || d(i-1)==0 || d(i)==0
              brk(end+1) = i; %#ok<AGROW>
          end
      end
      brk(end+1) = numel(wtab);

      % 汇总权重：Wsum = ∑ q^2/|vg|,  Vsum = ∑ q^2   ⇒  vg_eff = Vsum / Wsum
      Wsum = zeros(1, Nw);
      Vsum = zeros(1, Nw);

      for s = 1:numel(brk)-1
          i1 = brk(s); i2 = brk(s+1);
          if i2 - i1 < 2, continue; end

          qseg = qtab(i1:i2);
          wseg = wtab(i1:i2);

          % interp1 需要严格单调/严格增序，去重
          [wseg_u, iu] = unique(wseg, 'stable');
          qseg_u = qseg(iu);
          if numel(wseg_u) < 2, continue; end

          wlo = min(wseg_u);  whi = max(wseg_u);
          mask = (w_mid_1x >= wlo) & (w_mid_1x <= whi);
          if ~any(mask), continue; end

          wq = w_mid_1x(mask);
          q_of_w = interp1(wseg_u, qseg_u, wq, 'linear');    % q(w) 反插
          % v_g 在 q(w) 处（表插 + 线性）
          vg_at_q = abs(interp1(qtab, vgtab, q_of_w, 'linear', 'extrap'));
          vg_at_q = max(vg_at_q, 1e-6);                      % 避免 1/0

          q2 = q_of_w.^2;
          Wsum(mask) = Wsum(mask) + (q2 ./ vg_at_q);
          Vsum(mask) = Vsum(mask) + q2;
      end

      DOS_w_b(b,:) = (deg(b)/(2*pi^2)) * Wsum;         % 态密度计算DOS(q,b)=q^2/(2pi^2*vg)
      vg_w(b,:)    = zeros(1,Nw);
      nz = Wsum > 0;
      vg_w(b,nz)   = Vsum(nz) ./ Wsum(nz);             % 与 DOS 同权重的一致“有效 vg”
      % 其余保持 0
  end

  DOS_w = sum(DOS_w_b, 1);   % 总 DOS(ω)

  % ---------- 频率态的 Bose 占据与计数 ----------
  xw   = hbar .* w_mid_1x ./ (kB*T0);
  n_w1 = 1 ./ max(exp(min(xw,700)) - 1, realmin);      % 1×Nw

  % N_w^{b}(ω_j)Δω = DOS_w^{b}(ω_j) * n(ω_j) * Δω
  N_w = DOS_w_b .* repmat(n_w1 .* dw, B, 1);
  N_w_tot = sum(N_w, 'all');

  % ---------- 打包 spec ----------
  spec = struct();
  % q-网格
  spec.si        = mat;
  spec.T0        = T0;
  spec.q_edges   = q_edges;
  spec.q_mid     = q_mid;
  spec.dq        = dq;
  spec.qmax      = mat.qmax;

  spec.branches  = branches;
  spec.deg       = deg;
  spec.B         = B;
  spec.M         = M;

  spec.omega     = omega;            % B×M  （q_mid 处）
  spec.vg        = vg;               % B×M
  spec.Cv        = Cv;               % B×M
  spec.Cv_tot    = Cv_tot;

  spec.face_weight = face_weight;    % B×M
  spec.vol_weight  = vol_weight;     % B×M
  spec.cdf_face    = cdf_face;       % 1×(B*M)
  spec.cdf_vol     = cdf_vol;        % 1×(B*M)

  spec.bm2ind   = @(b,m) (b-1)*M + m;
  spec.ind2bm   = @(idx) deal( ceil(idx/M), 1+mod(idx-1,M) );
  spec.vg_max   = vg_max;

  % ω-网格
  spec.w_edges  = w_edges;           % 1×(Nw+1)
  spec.w_mid    = repmat(w_mid_1x, B, 1);  % B×Nw（按分支展开）
  spec.dw       = dw;                % 1×Nw
  spec.DOS_w    = DOS_w;             % 1×Nw
  spec.DOS_w_b  = DOS_w_b;           % B×Nw
  spec.N_w      = N_w;               % B×Nw
  spec.N_w_tot  = N_w_tot;           % 标量
  spec.vg_w     = vg_w;              % B×Nw（与 DOS 同权重的有效群速）
end
