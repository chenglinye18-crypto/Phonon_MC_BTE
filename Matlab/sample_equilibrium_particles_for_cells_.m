function [p, info] = sample_equilibrium_particles_for_cells_(mesh, spec, opts, cell_ids, Tcell, Tref_cell, id_start)
% sample_equilibrium_particles_for_cells_ Sample equilibrium particles for selected cells.

  if nargin < 7 || isempty(id_start)
      id_start = 0;
  end
  if nargin < 6
      Tref_cell = [];
  end

  cell_ids = double(cell_ids(:));
  Tcell = Tcell(:);
  Tref_cell = Tref_cell(:);

  info = struct('cell_ids', cell_ids, ...
                'Nexp_tot', 0, ...
                'Nsp_tot', 0, ...
                'target_temperature', Tcell, ...
                'reference_temperature', Tref_cell);
  p = repmat(local_blank_particle_(), 0, 1);

  if isempty(cell_ids)
      return;
  end
  if numel(Tcell) ~= numel(cell_ids)
      error('sample_equilibrium_particles_for_cells_: Tcell size must match cell_ids.');
  end
  if strcmpi(local_get_(opts, 'mode', 'absolute'), 'deviational') && numel(Tref_cell) ~= numel(cell_ids)
      error('sample_equilibrium_particles_for_cells_: Tref_cell size must match cell_ids in deviational mode.');
  end

  E_eff = local_get_(opts, 'E_eff', 1e-18);
  use_bin_center_w = local_get_(opts, 'use_bin_center_w', true);
  mode_name = local_get_(opts, 'mode', 'absolute');
  max_particles = local_get_(opts, 'max_particles', inf);

  [~, Vc_all] = local_cell_volumes_(mesh);
  af_all = local_enhance_factor_(opts, numel(Vc_all));
  Vc = Vc_all(cell_ids);
  af = af_all(cell_ids);

  kB = 1.380649e-23;
  hbar = 1.054571817e-34;
  [B, Nw] = size(spec.w_mid);
  [mode_weight, mode_sign] = local_cell_mode_spectra_(spec, mode_name, Tcell, Tref_cell, hbar, kB);

  mode_energy_abs = sum(mode_weight, 1).';
  cell_weight = mode_energy_abs .* Vc .* af;
  total_weight = sum(cell_weight);
  if total_weight <= 0
      return;
  end

  Nexp_tot = total_weight / E_eff;
  Nsp_tot = floor(Nexp_tot) + (rand < (Nexp_tot - floor(Nexp_tot)));
  info.Nexp_tot = Nexp_tot;
  info.Nsp_tot = Nsp_tot;
  if Nsp_tot <= 0
      return;
  end
  if Nsp_tot > max_particles
      error('sample_equilibrium_particles_for_cells_: expected particles %.3g exceed max_particles %.3g.', Nsp_tot, max_particles);
  end

  cdf_cell = cumsum(cell_weight) / total_weight;
  mode_cdf = zeros(B * Nw, numel(cell_ids));
  for iloc = 1:numel(cell_ids)
      if mode_energy_abs(iloc) > 0
          mode_cdf(:, iloc) = cumsum(mode_weight(:, iloc)) / mode_energy_abs(iloc);
      end
  end

  have_w_edges = isfield(spec, 'w_edges') && numel(spec.w_edges) == (Nw + 1);
  p = repmat(local_blank_particle_(), Nsp_tot, 1);

  for ip = 1:Nsp_tot
      iloc = local_sample_from_cdf_(cdf_cell);
      cid = cell_ids(iloc);
      [x, y, z] = local_uniform_position_in_cell_(mesh, cid);

      idx = local_sample_from_cdf_(mode_cdf(:, iloc));
      b = 1 + mod(idx - 1, B);
      m = floor((idx - 1) / B) + 1;

      if use_bin_center_w || ~have_w_edges
          w = spec.w_mid(b, m);
      else
          w = spec.w_edges(m) + (spec.w_edges(m + 1) - spec.w_edges(m)) * rand();
      end

      [q, vabs] = local_q_vabs_from_w_table_(w, spec, b);
      dir = local_rand_unit_vec_();

      sgn = +1;
      E = E_eff;
      if strcmpi(mode_name, 'deviational')
          sgn = mode_sign(idx, iloc);
          E = sgn * E_eff;
      end

      pid = id_start + ip;
      p(ip).id = pid;
      p(ip).par_id = pid;
      p(ip).cell = int32(cid);
      p(ip).x = x;
      p(ip).y = y;
      p(ip).z = z;
      p(ip).b = b;
      p(ip).m = m;
      p(ip).w = w;
      p(ip).q = q;
      p(ip).vabs = vabs;
      p(ip).v = vabs * dir;
      p(ip).E = E;
      p(ip).sgn = sgn;
      p(ip).n_ph = E / (hbar * max(w, 1e-30));
      p(ip).seed = randi(2^31 - 1);
      p(ip).t_left = 0;
  end
end

function [mode_weight, mode_sign] = local_cell_mode_spectra_(spec, mode_name, Tcell, Tref_cell, hbar, kB)
  [B, Nw] = size(spec.w_mid);
  Nc = numel(Tcell);

  if isvector(spec.dw)
      dw = repmat(reshape(spec.dw, 1, []), B, 1);
  else
      dw = spec.dw;
  end

  pref = hbar .* spec.w_mid .* max(spec.DOS_w_b, 0) .* dw;
  mode_weight = zeros(B * Nw, Nc);
  mode_sign = ones(B * Nw, Nc);

  for cid = 1:Nc
      n_cell = local_bose_(spec.w_mid, Tcell(cid), hbar, kB);
      if strcmpi(mode_name, 'deviational')
          n_ref = local_bose_(spec.w_mid, Tref_cell(cid), hbar, kB);
          Wbm = pref .* (n_cell - n_ref);
      else
          Wbm = pref .* n_cell;
      end

      mode_weight(:, cid) = abs(Wbm(:));
      if strcmpi(mode_name, 'deviational')
          sgn = sign(Wbm(:));
          sgn(sgn == 0) = +1;
          mode_sign(:, cid) = sgn;
      end
  end
end

function nBE = local_bose_(w, T, hbar, kB)
  T = max(T, 1e-12);
  x = hbar .* w ./ (kB * T);
  nBE = 1 ./ max(exp(min(x, 700)) - 1, realmin);
end

function af = local_enhance_factor_(opts, Nc)
  af = local_get_(opts, 'enhance_factor', 1);
  if isscalar(af)
      af = repmat(af, Nc, 1);
      return;
  end

  af = af(:);
  if numel(af) ~= Nc
      error('sample_equilibrium_particles_for_cells_: enhance_factor must be scalar or Nc-by-1.');
  end
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
  vabs = abs(interp1(qs, vs, q, 'pchip'));
end

function [x, y, z] = local_uniform_position_in_cell_(mesh, cid)
  epsl = 1e-12;
  [ix, iy, iz] = ind2sub([mesh.Nx, mesh.Ny, mesh.Nz], cid);
  X = mesh.x_edges(:);
  Y = mesh.y_edges(:);
  Z = mesh.z_edges(:);
  x = X(ix) + (X(ix + 1) - X(ix)) * rand();
  y = Y(iy) + (Y(iy + 1) - Y(iy)) * rand();
  z = Z(iz) + (Z(iz + 1) - Z(iz)) * rand();
  x = min(max(x, X(ix) + epsl), X(ix + 1) - epsl);
  y = min(max(y, Y(iy) + epsl), Y(iy + 1) - epsl);
  z = min(max(z, Z(iz) + epsl), Z(iz + 1) - epsl);
end

function idx = local_sample_from_cdf_(cdf)
  r = rand();
  idx = find(cdf >= r, 1, 'first');
  if isempty(idx)
      idx = numel(cdf);
  end
end

function u = local_rand_unit_vec_()
  u1 = rand();
  u2 = rand();
  cz = 2 * u1 - 1;
  sz = sqrt(max(0, 1 - cz^2));
  phi = 2 * pi * u2;
  u = [sz * cos(phi), sz * sin(phi), cz];
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

  error('sample_equilibrium_particles_for_cells_: mesh does not provide usable cell volumes.');
end

function s = local_blank_particle_()
  s = struct('id', 0, ...
             'cell', 0, ...
             'x', 0, 'y', 0, 'z', 0, ...
             'b', 0, 'm', 0, 'w', 0, 'q', 0, ...
             'v', [0 0 0], 'vabs', 0, ...
             'E', 0, 'sgn', +1, 'n_ph', 0, ...
             'seed', 0, 'par_id', 0, 't_left', 0);
end

function v = local_get_(s, name, default_v)
  if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
      v = s.(name);
  else
      v = default_v;
  end
end
