function newp = spawn_heat_source(opts, mesh, spec, state, Tprime, LUT, src, dt)
% spawn_heat_source Emit particles from a volumetric heat source region.
%
% Supported source fields:
%   src.type = 'volume'
%   src.qvol
%   src.E_eff
%   src.region

  if ~isfield(src, 'type') || ~strcmpi(src.type, 'volume')
      error('spawn_heat_source: only volumetric heat sources are supported.');
  end

  if ~isfield(src, 'E_eff') || isempty(src.E_eff)
      if isfield(opts, 'E_eff') && ~isempty(opts.E_eff)
          src.E_eff = opts.E_eff;
      else
          error('spawn_heat_source: src.E_eff or opts.E_eff is required.');
      end
  end

  newp = local_spawn_volume_Tloc_(opts, mesh, spec, state, Tprime, LUT, src, dt);
end

function p = local_spawn_volume_Tloc_(opts, mesh, spec, state, Tprime, LUT, src, dt)
  [cells, Vc, sample_pos_fun, ctr] = region_sampler_volume_(mesh, src);
  V_region = sum(Vc);
  Tloc = sample_T_at_point_from_Tprime_(mesh, Tprime, ctr);
  dE_tot = src.qvol * V_region * dt;
  Tsrc = invert_T_from_Udiff_(spec, LUT, Tloc, dE_tot / max(V_region, eps));
  Tref_loc = local_reference_temperature_for_region_(state, opts, cells, Tloc);

  Wbm = build_local_diff_spectrum_(spec, Tsrc, Tloc);
  if all(Wbm(:) == 0)
      p = struct([]);
      return;
  end

  next_id_base = get_next_id_(state);
  force_pos = isfield(src, 'force_positive') && src.force_positive;
  p = emit_particles_from_Wbm_refsample_( ...
      mesh, spec, cells, Vc, sample_pos_fun, ...
      Wbm, dE_tot, src.E_eff, Tref_loc, next_id_base, force_pos);
end

function Wbm = build_local_diff_spectrum_(spec, Tsrc, Tloc)
  kB = 1.380649e-23;
  hbar = 1.054571817e-34;
  bose = @(w, T) 1 ./ max(exp(min(hbar .* w ./ (kB * T), 700)) - 1, realmin);

  w = max(spec.w_mid, 0);
  DOS = max(spec.DOS_w_b, 0);
  if isvector(spec.dw)
      dw = repmat(reshape(spec.dw, 1, []), size(w, 1), 1);
  else
      dw = spec.dw;
  end

  n_src = bose(w, max(Tsrc, 1e-12));
  n_loc = bose(w, max(Tloc, 1e-12));
  Wbm = hbar .* w .* DOS .* (n_src - n_loc) .* dw;
  Wbm(DOS <= 0) = 0;
end

function Tsrc = invert_T_from_Udiff_(spec, LUT, Tloc, dU)
  if ~isempty(LUT) && isfield(LUT, 'inv') && isfield(LUT, 'T') && isfield(LUT, 'U')
      Uloc = interp1(LUT.T, LUT.U, Tloc, 'pchip', 'extrap');
      Tsrc = LUT.inv(Uloc + dU);
  else
      Tsrc = invert_monotone(@(T) U_density_equil(spec, T), U_density_equil(spec, Tloc) + dU, 1, 5000);
  end
end

function p = emit_particles_from_Wbm_refsample_(mesh, spec, cells, Vc, sample_pos_fun, ...
                                                Wbm, dE_tot, E_eff, Tref, next_id_base, force_positive)
  [B, Nw] = size(Wbm);
  Wabs = abs(Wbm);
  if sum(Wabs(:)) <= 0 || ~isfinite(sum(Wabs(:)))
      p = struct([]);
      return;
  end

  kB = 1.380649e-23;
  hbar = 1.054571817e-34;
  w = max(spec.w_mid, 0);
  DOS = max(spec.DOS_w_b, 0);
  if isvector(spec.dw)
      dw = repmat(reshape(spec.dw, 1, []), B, 1);
  else
      dw = spec.dw;
  end

  xref = (hbar .* w) / (kB * max(Tref, 1e-12));
  nref = 1 ./ max(exp(min(xref, 700)) - 1, realmin);
  Wref = hbar .* w .* DOS .* nref .* dw;
  if all(Wref(:) == 0)
      p = struct([]);
      return;
  end
  cdf_ref = cumsum(Wref(:)) / sum(Wref(:));
  nonzero_mask = (Wabs > 0);
  if ~any(nonzero_mask(:))
      p = struct([]);
      return;
  end

  Nexp = abs(dE_tot) / E_eff;
  Nsp = floor(Nexp) + (rand < (Nexp - floor(Nexp)));
  if Nsp == 0
      p = struct([]);
      return;
  end

  cdf_cell = cumsum(Vc(:)) / sum(Vc(:));
  p(Nsp, 1) = local_blank_particle_();
  has_edges = isfield(spec, 'w_edges') && numel(spec.w_edges) == (Nw + 1);

  for i = 1:Nsp
      [bb, mm] = local_pick_bm_nonzero_(cdf_ref, B, Nw, nonzero_mask);
      sgn = sign(Wbm(bb, mm));
      if sgn == 0
          sgn = +1;
      end
      if force_positive
          sgn = +1;
      end

      cid = cells(find(cdf_cell >= rand(), 1, 'first'));
      [x, y, z] = local_uniform_position_in_cell_(mesh, cid);

      if has_edges
          w_i = spec.w_edges(mm) + (spec.w_edges(mm + 1) - spec.w_edges(mm)) * rand();
      else
          w_i = w(bb, mm);
      end
      [q_i, vabs_i] = local_q_vabs_from_w_table_(w_i, spec, bb);
      dir = local_rand_unit_vec_();
      v_i = vabs_i * dir;

      pid = next_id_base + i;
      P = local_blank_particle_();
      P.id = pid;
      P.par_id = pid;
      P.cell = cid;
      P.x = x;
      P.y = y;
      P.z = z;
      P.b = bb;
      P.m = mm;
      P.w = w_i;
      P.q = q_i;
      P.v = v_i;
      P.vabs = vabs_i;
      P.E = sgn * E_eff;
      P.sgn = sgn;
      P.n_ph = (sgn * E_eff) / (hbar * max(w_i, 1e-30));
      P.seed = randi(2^31 - 1);
      p(i) = P;
  end
end

function [bb, mm] = local_pick_bm_nonzero_(cdf_ref, B, Nw, mask)
  for tries = 1:20
      r = rand();
      idx = find(cdf_ref >= r, 1, 'first');
      if isempty(idx)
          idx = numel(cdf_ref);
      end
      bb = 1 + mod(idx - 1, B);
      mm = floor((idx - 1) / B) + 1;
      if mask(bb, mm)
          return;
      end
  end
  [bb, mm] = find(mask, 1, 'first');
end

function [cells, Vc, sample_pos_fun, ctr] = region_sampler_volume_(mesh, src)
  if isfield(src, 'region') && isstruct(src.region)
      R = src.region;
  else
      R = struct('type', 'cells', 'id', []);
  end

  switch lower(R.type)
    case 'cells'
      [Nc, Vc_all] = local_cell_volumes_(mesh);
      if ~isfield(R, 'id') || isempty(R.id)
          cells = (1:Nc).';
      else
          cells = R.id(:);
      end
      Vc = Vc_all(cells);
      cdf = cumsum(Vc) / sum(Vc);
      sample_pos_fun = @() local_uniform_position_in_cell_(mesh, cells(find(cdf >= rand(), 1, 'first')));
      ctr = estimate_cells_center_(mesh, cells);
    case 'box'
      b = R.bounds;
      Vc = (b(2) - b(1)) * (b(4) - b(3)) * (b(6) - b(5));
      cells = [];
      sample_pos_fun = @() deal(b(1) + (b(2) - b(1)) * rand(), ...
                                b(3) + (b(4) - b(3)) * rand(), ...
                                b(5) + (b(6) - b(5)) * rand());
      ctr = [(b(1) + b(2)) / 2, (b(3) + b(4)) / 2, (b(5) + b(6)) / 2];
    case 'custom'
      cells = [];
      Vc = R.measure;
      sample_pos_fun = @() R.sample_pos();
      if isfield(R, 'center')
          ctr = R.center;
      else
          ctr = sample_pos_fun();
      end
    otherwise
      error('region_sampler_volume_: unsupported region.type');
  end
end

function ctr = estimate_cells_center_(mesh, cells)
  if all(isfield(mesh, {'x_edges', 'y_edges', 'z_edges', 'Nx', 'Ny', 'Nz'}))
      [ix, iy, iz] = ind2sub([mesh.Nx, mesh.Ny, mesh.Nz], round(mean(cells)));
      ctr = [mean(mesh.x_edges([ix ix + 1])), ...
             mean(mesh.y_edges([iy iy + 1])), ...
             mean(mesh.z_edges([iz iz + 1]))];
  elseif isfield(mesh, 'boxes')
      b = mesh.boxes(cells, :);
      c1 = mean(b(:, [1 3 5]), 1);
      c2 = mean(b(:, [2 4 6]), 1);
      ctr = 0.5 * (c1 + c2);
  else
      ctr = [0 0 0];
  end
end

function [Nc, Vc] = local_cell_volumes_(mesh)
  if isfield(mesh, 'cell_vol') && ~isempty(mesh.cell_vol)
      Vc = mesh.cell_vol(:);
      Nc = numel(Vc);
      return;
  end
  if all(isfield(mesh, {'x_edges', 'y_edges', 'z_edges', 'Nx', 'Ny', 'Nz'}))
      dx = diff(mesh.x_edges(:));
      dy = diff(mesh.y_edges(:));
      dz = diff(mesh.z_edges(:));
      [DX, DY, DZ] = ndgrid(dx, dy, dz);
      Vc = DX(:) .* DY(:) .* DZ(:);
      Nc = numel(Vc);
      return;
  end
  if isfield(mesh, 'boxes') && ~isempty(mesh.boxes)
      b = mesh.boxes;
      Vc = (b(:, 2) - b(:, 1)) .* (b(:, 4) - b(:, 3)) .* (b(:, 6) - b(:, 5));
      Nc = size(b, 1);
      return;
  end
  error('spawn_heat_source: mesh lacks cell volume information.');
end

function [x, y, z] = local_uniform_position_in_cell_(mesh, cid)
  epsl = 1e-12;
  if all(isfield(mesh, {'Nx', 'Ny', 'Nz', 'x_edges', 'y_edges', 'z_edges'}))
      [ix, iy, iz] = ind2sub([mesh.Nx, mesh.Ny, mesh.Nz], cid);
      X = mesh.x_edges;
      Y = mesh.y_edges;
      Z = mesh.z_edges;
      x = X(ix) + (X(ix + 1) - X(ix)) * rand();
      y = Y(iy) + (Y(iy + 1) - Y(iy)) * rand();
      z = Z(iz) + (Z(iz + 1) - Z(iz)) * rand();
      x = min(max(x, X(ix) + epsl), X(ix + 1) - epsl);
      y = min(max(y, Y(iy) + epsl), Y(iy + 1) - epsl);
      z = min(max(z, Z(iz) + epsl), Z(iz + 1) - epsl);
  elseif isfield(mesh, 'boxes')
      b = mesh.boxes(cid, :);
      x = b(1) + (b(2) - b(1)) * rand();
      y = b(3) + (b(4) - b(3)) * rand();
      z = b(5) + (b(6) - b(5)) * rand();
      x = min(max(x, b(1) + epsl), b(2) - epsl);
      y = min(max(y, b(3) + epsl), b(4) - epsl);
      z = min(max(z, b(5) + epsl), b(6) - epsl);
  else
      error('spawn_heat_source: mesh lacks geometry info to sample positions.');
  end
end

function Tloc = sample_T_at_point_from_Tprime_(mesh, Tprime, pt)
  cid = locate_cell_from_point_(mesh, pt);
  if cid > 0 && cid <= numel(Tprime)
      Tloc = Tprime(cid);
  else
      Tloc = mean(Tprime);
  end
end

function cid = locate_cell_from_point_(mesh, pt)
  x = pt(1);
  y = pt(2);
  z = pt(3);
  if all(isfield(mesh, {'x_edges', 'y_edges', 'z_edges', 'Nx', 'Ny', 'Nz'}))
      ix = find(mesh.x_edges <= x, 1, 'last');
      iy = find(mesh.y_edges <= y, 1, 'last');
      iz = find(mesh.z_edges <= z, 1, 'last');
      if isempty(ix) || isempty(iy) || isempty(iz) || ...
              ix >= numel(mesh.x_edges) || iy >= numel(mesh.y_edges) || iz >= numel(mesh.z_edges)
          cid = 0;
          return;
      end
      cid = sub2ind([mesh.Nx, mesh.Ny, mesh.Nz], ix, iy, iz);
  elseif isfield(mesh, 'boxes')
      cid = 0;
      bx = mesh.boxes;
      for i = 1:size(bx, 1)
          if x >= bx(i, 1) && x < bx(i, 2) && ...
                  y >= bx(i, 3) && y < bx(i, 4) && ...
                  z >= bx(i, 5) && z < bx(i, 6)
              cid = i;
              break;
          end
      end
  else
      cid = 0;
  end
end

function U = U_density_equil(spec, T)
  kB = 1.380649e-23;
  hbar = 1.054571817e-34;
  w = spec.w_mid;
  DOS = spec.DOS_w_b;
  if isvector(spec.dw)
      dw = repmat(reshape(spec.dw, 1, []), size(w, 1), 1);
  else
      dw = spec.dw;
  end
  n = 1 ./ max(exp(min(hbar .* w ./ (kB * T), 700)) - 1, realmin);
  U = sum(sum(hbar .* w .* DOS .* n .* dw));
end

function Tsol = invert_monotone(fun, target, Tmin, Tmax, tol)
  if nargin < 5
      tol = 1e-6;
  end
  fmin = fun(Tmin) - target;
  fmax = fun(Tmax) - target;
  if fmin > 0
      error('invert_monotone: lower bound too high');
  end
  if fmax < 0
      error('invert_monotone: upper bound too low');
  end
  for k = 1:100
      Tmid = 0.5 * (Tmin + Tmax);
      fmid = fun(Tmid) - target;
      if abs(fmid) <= max(1e-12, tol * max(1, abs(target)))
          Tsol = Tmid;
          return;
      end
      if fmid > 0
          Tmax = Tmid;
      else
          Tmin = Tmid;
      end
  end
  Tsol = 0.5 * (Tmin + Tmax);
end

function nid = get_next_id_(state)
  if isempty(state.p)
      nid = 0;
  else
      ids = [state.p.id];
      if isempty(ids)
          nid = 0;
      else
          nid = max(ids);
      end
  end
end

function s = local_blank_particle_()
  s = struct( ...
      'id', 0, ...
      'cell', 0, ...
      'x', 0, 'y', 0, 'z', 0, ...
      'b', 0, 'm', 0, 'w', 0, 'q', 0, ...
      'v', [0 0 0], 'vabs', 0, ...
      'E', 0, 'sgn', +1, 'n_ph', 0, ...
      'seed', 0, 'par_id', 0, 't_left', 0);
end

function [q, vabs] = local_q_vabs_from_w_table_(w, spec, b)
  qv = spec.si.q(:);
  wv = spec.si.omega_tab(b, :).';
  gv = spec.si.vg_tab(b, :).';

  [w_sorted, Is] = sort(max(wv, 0), 'ascend');
  q_sorted = qv(Is);
  v_sorted = gv(Is);
  [ws, Iu] = unique(w_sorted, 'stable');
  qs = q_sorted(Iu);
  vs = v_sorted(Iu);

  w_cl = min(max(w, ws(1)), ws(end));
  q = interp1(ws, qs, w_cl, 'pchip');
  v = interp1(qs, vs, q, 'pchip');
  vabs = abs(v);
end

function dir = local_rand_unit_vec_()
  u1 = rand();
  u2 = rand();
  cz = 2 * u1 - 1;
  theta = 2 * pi * u2;
  sx = sqrt(1 - cz ^ 2);
  dir = [sx * cos(theta), sx * sin(theta), cz];
end

function Tref_loc = local_reference_temperature_for_region_(state, opts, cells, fallback_T)
  if nargin < 4 || isempty(fallback_T)
      fallback_T = 300;
  end

  if isstruct(state) && isfield(state, 'info') && isstruct(state.info) && ...
          isfield(state.info, 'Tref_cell') && ~isempty(state.info.Tref_cell)
      Tref_all = state.info.Tref_cell(:);
      if ~isempty(cells)
          Tref_loc = mean(Tref_all(cells));
      else
          Tref_loc = mean(Tref_all);
      end
      return;
  end

  if isfield(opts, 'Tref_cell') && ~isempty(opts.Tref_cell)
      Tref_all = opts.Tref_cell(:);
      if ~isempty(cells)
          Tref_loc = mean(Tref_all(cells));
      else
          Tref_loc = mean(Tref_all);
      end
      return;
  end

  if isfield(opts, 'Tref') && isfinite(opts.Tref)
      Tref_loc = opts.Tref;
  else
      Tref_loc = fallback_T;
  end
end
