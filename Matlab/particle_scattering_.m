function [state, after_energy] = particle_scattering_(state, mesh, spec, opts, dt, Tstar, r_tau)
% PARTICLE_SCATTERING_
%  - PI: 只改方向，不改频率/分支/能量
%  - PP: 用  ħω·D·Γpp(·|Tcell)·W(ω)  在 (branch, ω) 上重采样
%        absolute   : W = nB(ω, Tcell)
%        deviational: W = |nB(ω, Tloc_cell) - nB(ω, Tref)|
%        命中后固定能量大小 E_eff；偏差法的“符号”= sign(ΔnB) 于被抽中 (b,m) 处
%  - PB: 默认关闭（保留开关 opts.scatter.pb_on）
%
% 输入：
%   state.p(i): .cell(>0), .b, .w, .vabs, .v(1×3), .E(=±E_eff), 可选 .sgn
%   spec: w_mid(B×Nw), 可选 w_edges(1×Nw+1), DOS_w_b(B×Nw)
%         q/|vg| 查表需：spec.si.q(:), spec.si.omega_tab(b,:), spec.si.vg_tab(b,:)
%   opts.mode = 'absolute' | 'deviational'（默认 'absolute'）
%   opts.Tref（仅 deviational 需要）
%   opts.Tloc_cell（可选, Nc×1）：若提供则严格使用式(23)；未提供时用 Tcell 近似
%   opts.scatter.pb_on = false（默认）
%   dt, Tstar(Nc×1) —— cell 温度
%   r_tau(Np×6): [LA, TAN, TAU, LTO, PI, PB]
%
% 输出：
%   state: 更新后的粒子
%   after_energy(Nc×1): 本步散射后每 cell 的总能量（按 state.p(i).E 聚合，有符号）

  if nargin < 7, error('particle_scattering_: need state, mesh, spec, opts, dt, Tstar, r_tau'); end
  if ~isfield(opts,'mode'), opts.mode = 'absolute'; end
  if ~isfield(opts,'scatter') || ~isstruct(opts.scatter), opts.scatter = struct(); end
  pb_on = isfield(opts.scatter,'pb_on') && logical(opts.scatter.pb_on);

  Np = numel(state.p);
  Nc = infer_Nc_(mesh);
  if Np==0, after_energy = zeros(Nc,1); return; end

  % ------- 拉基础数组 -------
  cell_id = int32([state.p.cell].');                       % Np×1
  b       = int32([state.p.b].');
  w       = [state.p.w].';
  vabs    = [state.p.vabs].';
  v       = vertcat(state.p.v);

  % 记录“散射前”的符号（用于开发期自检；不会被后续修改）
  sgn_before = particle_sign_vec_(state, opts);            % Np×1, ∈{+1,-1}

  % ------- 拉基础数组 -------
  cell_id = int32([state.p.cell].');                       % Np×1
  b       = int32([state.p.b].');
  w       = [state.p.w].';
  vabs    = [state.p.vabs].';
  v       = vertcat(state.p.v);
  E_vec   = get_particle_energy_vec_(state, opts);         % Np×1（有符号）

  % ------- 速率向量拆分 -------
  rLA  = max(r_tau(:,1),0);
  rTAN = max(r_tau(:,2),0);
  rTAU = max(r_tau(:,3),0);
  rLTO = max(r_tau(:,4),0);
  rPI  = max(r_tau(:,5),0);
  rPB  = max(r_tau(:,6),0);
  if ~pb_on, rPB(:) = 0; end

  % PP 合成（按分支）
  rPP = zeros(Np,1);
  rPP(b==1) = rLA(b==1);                   % LA
  rPP(b==2) = rTAN(b==2) + rTAU(b==2);     % TA
  rPP(b> 2) = rLTO(b>2);                   % LO/TO

  rTOT = rPP + rPI + rPB;

  % ------- 是否散射 -------
  Ptot = 1 - exp(-dt .* max(rTOT,0));
  hit  = rand(Np,1) < Ptot;

  % ------- 在命中者中分配类型 -------
  sel = rand(nnz(hit),1) .* rTOT(hit);
  rPP_hit = rPP(hit); rPI_hit = rPI(hit);

  isPP = false(Np,1);
  isPI = false(Np,1);
  isPB = false(Np,1);

  isPP(hit) = (sel < rPP_hit);
  isPI(hit) = (~isPP(hit)) & (sel < (rPP_hit + rPI_hit));
  isPB(hit) = ~isPP(hit) & ~isPI(hit);   % pb_on=false 时一般为0
  changed = isPP | isPI | (pb_on & isPB);

  % ===================== PI：只改方向 =====================
  if any(isPI)
      nPI = nnz(isPI);
      [ex,ey,ez] = rand_unit_vec_batch_(nPI);
      vi = find(isPI);
      v(vi,1) = vabs(vi).*ex;
      v(vi,2) = vabs(vi).*ey;
      v(vi,3) = vabs(vi).*ez;
  end

  % ===================== PB：默认关闭 =====================
  if pb_on && any(isPB)
      % 可按需仿 PI 改向
  end

  % ===================== PP：按 Γpp(b,ω|Tcell) × 目标谱 重采样 =====================
  if any(isPP)
      iiPP  = find(isPP);
      cPP   = double(cell_id(iiPP));
      Tref_cell = local_reference_temperature_field_(state, opts, Nc);
      B     = size(spec.w_mid, 1);
      Nw    = size(spec.w_mid, 2);
      w_edges = get_w_edges_(spec);                 % 1×(Nw+1)
      dw      = reshape(w_edges(2:end)-w_edges(1:end-1), 1, Nw);

      Gamma_fallback = build_pp_rate_table_from_particles_(b, w, rPP, w_edges, B, Nw);

      [uc,~,ic] = unique(cPP);
      nGroups = numel(uc);

      b_new    = zeros(numel(iiPP),1,'int32');
      w_new    = zeros(numel(iiPP),1);
      vabs_new = zeros(numel(iiPP),1);

      use_center = ~isfield(opts,'use_bin_center_w') || opts.use_bin_center_w;
      kB   = 1.380649e-23; 
      hbar = 1.054571817e-34;

      have_Tloc = isfield(opts,'Tloc_cell') && numel(opts.Tloc_cell)==Nc;

      for g = 1:nGroups
          cid   = uc(g);
          Tcell = max(Tstar(min(max(cid,1),Nc)), 1e-12);
          if have_Tloc, Tloc = max(opts.Tloc_cell(min(max(cid,1),Nc)), 1e-12); else, Tloc = Tcell; end

          Gamma = pp_rate_table_at_T_(spec, Tcell, opts, Gamma_fallback);  % B×Nw, ≥0
          if ~any(Gamma(:)>0), Gamma = Gamma_fallback; end

          % DOS & 目标谱权重 W（与原版一致）
          if isfield(spec,'DOS_w_b') && ~isempty(spec.DOS_w_b)
              DOS = max(spec.DOS_w_b,0);
          elseif isfield(spec,'N_w') && ~isempty(spec.N_w)
              DOS = max(spec.N_w,0) ./ repmat(dw, size(spec.w_mid,1), 1);
          else
              DOS = ones(size(spec.w_mid));
          end
          wmid = spec.w_mid;

          if ~isfield(opts,'mode') || strcmpi(opts.mode,'absolute')
              x    = hbar .* wmid ./ (kB*Tcell);
              nBE  = 1 ./ max(exp(min(x,700)) - 1, realmin);
              W    = (hbar .* wmid) .* DOS .* nBE .* Gamma .* repmat(dw, B, 1);
          else
              % 偏差法：仍按 |ΔnB| 选频带，但【不再】改能量符号
              Tref_loc = max(Tref_cell(min(max(cid,1),Nc)), 1e-12);
              x_ref = hbar .* wmid ./ (kB*Tref_loc);
              n_ref = 1 ./ max(exp(min(x_ref,700)) - 1, realmin);
              x_loc = hbar .* wmid ./ (kB*Tloc);
              n_loc = 1 ./ max(exp(min(x_loc,700)) - 1, realmin);
              dNB   = n_loc - n_ref;                      % 仅用于构造权重
              W     = (hbar .* wmid) .* DOS .* abs(dNB) .* Gamma .* repmat(dw, B, 1);
          end

          W(~isfinite(W)) = 0;
          if ~any(W(:)>0), W(:)=1; end

          % —— CDF & 抽样 —— 
          Wlin = W(:);
          cdf  = cumsum(Wlin) / sum(Wlin);

          loc  = (ic == g);
          nloc = nnz(loc);
          rsel = rand(nloc,1);
          lin_idx = arrayfun(@(r)find(cdf>=r,1,'first'), rsel);
          lin_idx(isnan(lin_idx)) = numel(Wlin);  % 兜底

          % 正确的 column-major 反解：lin → (b,m)
          b_pick = int32(1+mod(lin_idx-1, B));
          m_pick = floor((lin_idx-1)/B)+1;

          % 频率
          if use_center || ~isfield(spec,'w_edges') || numel(spec.w_edges)~=(Nw+1)
              w_pick = spec.w_mid(sub2ind([B,Nw], b_pick, m_pick));
          else
              w_lo = w_edges(m_pick); w_hi = w_edges(m_pick+1);
              w_pick = w_lo + (w_hi - w_lo).*rand(nloc,1);
          end

          % 查表回填 |vg|
          vabs_loc = zeros(nloc,1);
          for kk = 1:nloc
              [~, vgk] = q_vabs_from_w_table_(w_pick(kk), spec, b_pick(kk));
              vabs_loc(kk) = vgk;
          end

          % 写入组位置（仅态变量）
          b_new(loc)    = b_pick;
          w_new(loc)    = w_pick;
          vabs_new(loc) = vabs_loc;

          % ⚠️ 不再改 state.p(i).E 或 state.p(i).sgn —— 符号保持不变
      end

      % 新方向各向同性
      [ex,ey,ez] = rand_unit_vec_batch_(numel(iiPP));
      v_new = [vabs_new.*ex, vabs_new.*ey, vabs_new.*ez];

      % 回填到全体数组
      b(iiPP)    = b_new;
      w(iiPP)    = w_new;
      vabs(iiPP) = vabs_new;
      v(iiPP,:)  = v_new;
  end

  % （可选）开发期自检：符号未改变
  sgn_after = particle_sign_vec_(state, opts);
  % 只检查 changed 的粒子，若失败给出 warning（改成 assert 也行）
  if any(changed)
      if any(sgn_before(changed) ~= sgn_after(changed))
          warning('particle_scattering_: sgn changed on some particles, which should not happen.');
      end
  end

  % ------- 写回 state（仅写变动者）-------
  changed = isPP | isPI | (pb_on & isPB);
  if any(changed)
      idxc = find(changed);
      for k = 1:numel(idxc)
          i = idxc(k);
          state.p(i).b    = b(i);
          state.p(i).w    = w(i);
          state.p(i).vabs = vabs(i);
          state.p(i).v    = v(i,:);
          % ⚠️ 不改 state.p(i).E / state.p(i).sgn
      end
  end

  % （可选）开发期自检：符号未改变（只在 changed 时检查）
  if any(changed)
      sgn_after = particle_sign_vec_(state, opts);
      if any(sgn_before(changed) ~= sgn_after(changed))
          warning('particle_scattering_: sgn changed on some particles, which should not happen.');
      end
  end

  % ------- after_energy：每 cell 聚合（有符号） -------
  valid = (cell_id >= 1) & (cell_id <= Nc);
  if any(valid)
      E_use = [state.p.E].';
      after_energy = accumarray(double(cell_id(valid)), E_use(valid), [Nc 1], @sum, 0);
  else
      after_energy = zeros(Nc,1);
  end

  % ------- 打印 -------
  n_hit = nnz(hit);
  n_PI  = nnz(isPI);
  n_PP  = nnz(isPP);
  n_PB  = nnz(isPB);
  E_tot = sum([state.p.E]);
  fprintf(1,'        [scatter] N=%d | hits=%d (PP=%d, PI=%d, PB=%d) | E_total=%.3e J\n', ...
      Np, n_hit, n_PP, n_PI, n_PB, E_tot);
end

% ====================== 工具：有符号能量向量 ======================
function E = get_particle_energy_vec_(state, opts)
  if ~isempty(state.p) && isfield(state.p(1),'E') && ~isempty([state.p.E])
      E = [state.p.E].';
  else
      E_eff = 1e-18;
      if isfield(opts,'E_eff') && isfinite(opts.E_eff) && opts.E_eff>0
          E_eff = opts.E_eff;
      end
      if isfield(state.p,'sgn')
          sgn = [state.p.sgn].'; sgn(sgn==0) = +1;
      else
          sgn = ones(numel(state.p),1);
      end
      E = sgn .* E_eff;
  end
end

% ===== 小工具：粒子符号向量 =====
function s = particle_sign_vec_(state, opts)
  if ~isempty(state.p) && isfield(state.p(1),'E') && ~isempty([state.p.E])
      s = sign([state.p.E].'); s(s==0) = +1;
  elseif isfield(state.p,'sgn') && ~isempty([state.p.sgn])
      s = [state.p.sgn].'; s(s==0) = +1;
  else
      s = ones(numel(state.p),1); % 无字段时默认 +1
  end
end

% ====================== Γ_pp(b,ω|T) 表（优先用你提供的） ======================
function Gamma = pp_rate_table_at_T_(spec, T, opts, fallback_table)
  B = size(spec.w_mid,1); Nw = size(spec.w_mid,2);
  if isfield(opts,'pp_rate_table_fun') && isa(opts.pp_rate_table_fun,'function_handle')
      Gamma = opts.pp_rate_table_fun(T, spec);
  elseif isfield(spec,'pp_rate_table_fun') && isa(spec.pp_rate_table_fun,'function_handle')
      Gamma = spec.pp_rate_table_fun(T);
  elseif isfield(spec,'Gamma_pp_T') && ~isempty(spec.Gamma_pp_T) ...
      && ndims(spec.Gamma_pp_T)==3 && size(spec.Gamma_pp_T,1)==B && size(spec.Gamma_pp_T,2)==Nw ...
      && isfield(spec,'T_grid') && numel(spec.T_grid)==size(spec.Gamma_pp_T,3)
      Tg = spec.T_grid(:); G3 = spec.Gamma_pp_T;
      if T <= Tg(1), Gamma = G3(:,:,1);
      elseif T >= Tg(end), Gamma = G3(:,:,end);
      else
          k = find(Tg<=T,1,'last'); a = (T-Tg(k))/(Tg(k+1)-Tg(k));
          Gamma = (1-a)*G3(:,:,k) + a*G3(:,:,k+1);
      end
  else
      Gamma = fallback_table;
  end
  if isempty(Gamma), Gamma = zeros(B,Nw); end
  Gamma(~isfinite(Gamma)) = 0; Gamma = max(Gamma,0);
  if ~any(Gamma(:)>0), Gamma = max(fallback_table,0); if ~any(Gamma(:)>0), Gamma(:)=1; end, end
end

% ====================== 由当前粒子构建全局平均 Γpp（兜底） ======================
function Gamma_tab = build_pp_rate_table_from_particles_(b_vec, w_vec, rPP_vec, w_edges, B, Nw)
  m_vec = discretize(w_vec, w_edges); m_vec(~isfinite(m_vec)) = 0; m_vec = max(1, min(Nw, m_vec));
  lin = sub2ind([B, Nw], double(b_vec), double(m_vec));
  sumR = accumarray(lin, rPP_vec, [B*Nw, 1], @sum, 0);
  cntR = accumarray(lin, 1,       [B*Nw, 1], @sum, 0);
  Sum = reshape(sumR, [B, Nw]); Cnt = reshape(cntR, [B, Nw]);
  Gamma_tab = zeros(B,Nw); mask = Cnt > 0; Gamma_tab(mask) = Sum(mask) ./ Cnt(mask);
  global_mean = sum(Sum(:)) / max(sum(Cnt(:)), 1); if ~isfinite(global_mean) || global_mean<=0, global_mean = 1e-30; end
  for bb = 1:B
      if any(mask(bb,:)), row_mean = sum(Sum(bb,:)) / max(sum(Cnt(bb,:)),1); fill_val = row_mean;
          if ~(isfinite(fill_val) && fill_val>0), fill_val = global_mean; end
      else, fill_val = global_mean; end
      zero_pos = (Cnt(bb,:)==0); Gamma_tab(bb,zero_pos) = fill_val;
  end
  if ~any(Gamma_tab(:)>0), Gamma_tab(:)=1; end
end

% ====================== 频率边界 ======================
function w_edges = get_w_edges_(spec)
  if isfield(spec,'w_edges') && numel(spec.w_edges) == size(spec.w_mid,2)+1
      w_edges = spec.w_edges(:).';
  else
      wm = spec.w_mid(1,:); w_edges = zeros(1, numel(wm)+1);
      w_edges(2:end-1) = 0.5*(wm(1:end-1)+wm(2:end));
      dw1 = wm(2)-wm(1); dwn = wm(end)-wm(end-1);
      w_edges(1)   = wm(1)   - max(dw1, 1e-30);
      w_edges(end) = wm(end) + max(dwn, 1e-30);
  end
end

% ====================== 随机各向同性方向（批量） ======================
function [ex,ey,ez] = rand_unit_vec_batch_(N)
  u1 = rand(N,1); u2 = rand(N,1);
  cz = 2*u1 - 1; sz = sqrt(max(0,1 - cz.^2)); phi = 2*pi*u2;
  ex = sz.*cos(phi); ey = sz.*sin(phi); ez = cz;
end

% ====================== 查表 (w,b) → q, |vg| ======================
function [q, vabs] = q_vabs_from_w_table_(w, spec, b)
  qv = spec.si.q(:); wv = spec.si.omega_tab(b,:).'; gv = spec.si.vg_tab(b,:).';
  [w_sorted, Is] = sort(wv, 'ascend'); q_sorted = qv(Is); v_sorted = gv(Is);
  [ws, Iu]  = unique(w_sorted, 'stable'); qs = q_sorted(Iu); vs = v_sorted(Iu);
  w = w(:); w_cl = min(max(w, ws(1)), ws(end));
  q  = interp1(ws, qs, w_cl, 'pchip'); vg = interp1(qs, vs, q, 'pchip');
  vabs = abs(vg); q = reshape(q, size(w_cl)); vabs = reshape(vabs, size(w_cl));
end

% ====================== 推断 Nc ======================
function Nc = infer_Nc_(mesh)
  if isfield(mesh,'Nc') && ~isempty(mesh.Nc), Nc = mesh.Nc; return; end
  if all(isfield(mesh,{'Nx','Ny','Nz'})) && ~isempty(mesh.Nx), Nc = mesh.Nx*mesh.Ny*mesh.Nz; return; end
  if isfield(mesh,'cell_vol') && ~isempty(mesh.cell_vol), Nc = numel(mesh.cell_vol); return; end
  if isfield(mesh,'boxes') && ~isempty(mesh.boxes), Nc = size(mesh.boxes,1); return; end
  error('particle_scattering_: 无法从 mesh 推断 Nc');
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

  error('particle_scattering_: deviational 模式需提供 Tref_cell 或 opts.Tref.');
end
