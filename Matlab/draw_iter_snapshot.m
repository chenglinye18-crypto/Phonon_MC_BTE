function draw_iter_snapshot(state, spec, mesh, iter_idx, opts)
% 每轮迭代开始时绘图快照（带基准能量叠加的分支能量与分支能量谱）
% 图像：
%   (1) 实空间散点（离散分支着色）
%   (2) k 空间散点（离散分支着色，k = q·v̂）
%   (3) 分支总能量（物理能量 = 基准 + 偏差）
%   (4) 分支能量谱 |E_b(ω)|，log 纵轴，DOS=0 处断线
%
% 需要：
%   spec.DOS_w_b (B×Nw), spec.w_mid (B×Nw), spec.dw (1×Nw)
%   若 state.info.mode == 'deviational' 则需 Tref（opts.Tref 或 state.info.Tref）

  if nargin<5, opts = struct(); end
  if isfield(opts, 'viz') && isstruct(opts.viz) ...
          && isfield(opts.viz, 'enable') && ~opts.viz.enable
      return;
  end
  if ~isfield(opts,'max_points'), opts.max_points = 5e4; end
  if ~isfield(opts,'marker_sz'),  opts.marker_sz  = 6;   end
  if ~isfield(opts,'alpha'),      opts.alpha      = 0.6; end
  if ~isfield(opts,'save_path'),  opts.save_path  = '';  end
  if ~isfield(opts,'figure_id'),  opts.figure_id  = 9021; end

  p = state.p;
  if isempty(p), warning('draw_iter_snapshot: state.p 为空，跳过绘图。'); return; end

  % ---------- 常量 ----------
  kB   = 1.380649e-23;
  hbar = 1.054571817e-34;
  bose = @(w,T) 1 ./ max(exp(min(hbar.*w./(kB*T),700)) - 1, realmin);
  
  % ---------- 尺寸/分支名 ----------
  B  = size(spec.w_mid,1);
  Nw = size(spec.w_mid,2);
  
  if isfield(spec,'br_names') && ~isempty(spec.br_names)
      br_names = spec.br_names;
  elseif isfield(spec,'b_names') && ~isempty(spec.b_names)
      br_names = spec.b_names;
  else
      canon = {'TA','LA','TO','LO'};
      br_names = cell(1,B);
      for bb = 1:B
          if bb <= numel(canon)
              br_names{bb} = canon{bb};
          else
              br_names{bb} = sprintf('B%d', bb);
          end
      end
  end
  if isstring(br_names), br_names = cellstr(br_names); end


  % ---------- 实空间范围 ----------
  has_edges = all(isfield(mesh,{'x_edges','y_edges','z_edges'}));
  if has_edges
      xlimv = [mesh.x_edges(1) mesh.x_edges(end)];
      ylimv = [mesh.y_edges(1) mesh.y_edges(end)];
      zlimv = [mesh.z_edges(1) mesh.z_edges(end)];
  else
      xs_all = [p.x]; ys_all = [p.y]; zs_all = [p.z];
      xlimv = [min(xs_all) max(xs_all)];
      ylimv = [min(ys_all) max(ys_all)];
      zlimv = [min(zs_all) max(zs_all)];
  end

  % ---------- 下采样（仅可视化） ----------
  N = numel(p);
  if N > opts.max_points, sel = randperm(N, opts.max_points); else, sel = 1:N; end

  xs = [p(sel).x]; ys = [p(sel).y]; zs = [p(sel).z];
  vs = vertcat(p(sel).v);
  vabs = [p(sel).vabs]; vabs(vabs==0) = 1e-30;
  qs = [p(sel).q];
  b_all = [p.b];               b_sel = b_all(sel);
  m_all = [p.m];
  E_all = [p.E];

  % ---------- k 向量（v̂ 方向） ----------
  vunit = vs ./ vabs(:);
  kvec  = vunit .* qs(:);
  kx = kvec(:,1); ky = kvec(:,2); kz = kvec(:,3);

  % ---------- 频率轴 ----------
  if isfield(spec,'w_edges') && numel(spec.w_edges)==(Nw+1)
      w_edges   = spec.w_edges(:).';
      w_centers = 0.5*(w_edges(1:end-1)+w_edges(2:end));
  else
      w_centers = mean(spec.w_mid,1);
  end
  dw = spec.dw(:).';  % 1×Nw

  % ---------- DOS 掩膜 ----------
  hasDOS_all = true(B,Nw);
  if isfield(spec,'DOS_w_b') && ~isempty(spec.DOS_w_b)
      hasDOS_all = spec.DOS_w_b > 0;
  end

  % ---------- 偏差能量按 (b,m) 聚合 ----------
  % dev_E_bm(b,m)：由粒子贡献的偏差能量（可正负）
  dev_E_bm = zeros(B,Nw);
  % 使用 accumarray 聚合到 Nw×B 再转置到 B×Nw
  dev_E_m_b = accumarray([m_all(:), b_all(:)], E_all(:), [Nw, B], @sum, 0);
  dev_E_bm  = dev_E_m_b.';  % B×Nw

  % ---------- 基准能量（只在 deviational 模式下叠加） ----------
  add_baseline = true;
  mode_is_dev  = isfield(state,'info') && isfield(state.info,'mode') && strcmpi(state.info.mode,'deviational');
  if mode_is_dev && add_baseline
      % 获取 Tref
      Tref = opts.Tref;
      w_mid = spec.w_mid;            % B×Nw
      DOSb  = spec.DOS_w_b;          % B×Nw
      nref  = bose(w_mid, Tref);     % B×Nw
      base_E_bm = hbar .* w_mid .* DOSb .* nref .* repmat(dw, B, 1);  % B×Nw
  else
      base_E_bm = zeros(B,Nw);
  end

  % ---------- 物理能量：基准 + 偏差 ----------
  phys_E_bm = base_E_bm + dev_E_bm;        % B×Nw
  % 分支总能量（对 m 求和）
  E_branch_total = sum(phys_E_bm, 2);      % B×1

  % ---------- 绘图 ----------
  f = figure(opts.figure_id); clf(f); set(f,'Color','w');
  tl = tiledlayout(f, 2, 2, 'TileSpacing','compact', 'Padding','compact');
  title(tl, sprintf('MC-BTE Snapshot @ iter %d  (N=%d, shown=%d)', iter_idx, N, numel(sel)));

  % (1) Real space
  ax1 = nexttile(tl,1);
  scatter3(ax1, xs, ys, zs, opts.marker_sz, double(b_sel), 'filled', 'MarkerFaceAlpha', opts.alpha);
  axis(ax1,'equal'); grid(ax1,'on'); box(ax1,'on');
  xlim(ax1,xlimv); ylim(ax1,ylimv); zlim(ax1,zlimv);
  xlabel(ax1,'x'); ylabel(ax1,'y'); zlabel(ax1,'z'); title(ax1,'Real space');
  colormap(ax1, lines(B)); caxis(ax1,[0.5, B+0.5]);
  cb1 = colorbar(ax1); cb1.Ticks = 1:B; cb1.TickLabels = br_names; cb1.Label.String = 'branch';

  % (2) k space
  ax2 = nexttile(tl,2);
  scatter3(ax2, kx, ky, kz, opts.marker_sz, double(b_sel), 'filled', 'MarkerFaceAlpha', opts.alpha);
  axis(ax2,'equal'); grid(ax2,'on'); box(ax2,'on');
  xlabel(ax2,'k_x'); ylabel(ax2,'k_y'); zlabel(ax2,'k_z'); title(ax2,'k space (k = q·v̂)');
  colormap(ax2, lines(B)); caxis(ax2,[0.5, B+0.5]);
  cb2 = colorbar(ax2); cb2.Ticks = 1:B; cb2.TickLabels = br_names; cb2.Label.String = 'branch';

  % (3) Branch total energy (baseline + deviation)
  ax3 = nexttile(tl,3);
  bar(ax3, 1:B, E_branch_total, 'FaceAlpha',0.9);
  grid(ax3,'on'); box(ax3,'on');
  xticks(ax3, 1:B); xticklabels(ax3, br_names);
  xlabel(ax3,'branch'); ylabel(ax3,'energy (J)');
  title(ax3,'Branch energy');

  % (4) Energy spectrum by branch
  ax4 = nexttile(tl,4); hold(ax4,'on');
  cmap = lines(B);
  for b = 1:B
      y = phys_E_bm(b,:);             % 1×Nw
      mask = hasDOS_all(b,:);         % 1×Nw
      y(~mask) = NaN;                 % DOS=0 的 bin 断线
      y(y<=0)  = NaN;                 % log 纵轴不能显示 <=0
      plot(ax4, w_centers(:), y(:), '-', 'LineWidth',1.3, 'Color', cmap(b,:));
  end
  set(ax4,'YScale','log'); grid(ax4,'on'); box(ax4,'on'); hold(ax4,'off');
  xlabel(ax4,'\omega  (rad/s)'); ylabel(ax4,'energy per bin  (J)');
  title(ax4,'Energy spectrum by branch');
  legend(ax4, br_names, 'Location','best');

  % 保存
  if ~isempty(opts.save_path)
      if ~exist(opts.save_path,'dir'), mkdir(opts.save_path); end
      fname = fullfile(opts.save_path, sprintf('snapshot_iter_%05d.png', iter_idx));
      exportgraphics(f, fname, 'Resolution', 160);
  end
end
