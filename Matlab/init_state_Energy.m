function state = init_state_Energy(mesh, spec, opts)
% init_state_Energy Initialize deviational/absolute particles from per-cell fields.
%
% Supported options:
%   opts.mode = 'absolute' | 'deviational'
%   opts.initial_temperature_file
%   opts.reference_temperature_file
%   opts.T_init          scalar fallback for absolute mode
%   opts.T0              scalar fallback / linearization temperature
%   opts.Tref            reference temperature for deviational mode
%   opts.E_eff
%   opts.use_bin_center_w
%   opts.enhance_factor
%   opts.max_particles
%
% Inputs are fully file-backed now:
%   - Tcell comes from initial_temperature_file
%   - Tref_cell comes from reference_temperature_file
% The initializer stores both fields into state.info so later temperature update,
% scattering, and reservoir refresh can reuse the same reference data.

  if nargin < 3, opts = struct(); end
  if ~isfield(opts, 'mode') || isempty(opts.mode), opts.mode = 'absolute'; end
  if ~isfield(opts, 'E_eff') || isempty(opts.E_eff), opts.E_eff = 1e-18; end
  if ~isfield(opts, 'use_bin_center_w') || isempty(opts.use_bin_center_w), opts.use_bin_center_w = true; end
  if ~isfield(opts, 'enhance_factor') || isempty(opts.enhance_factor), opts.enhance_factor = 1; end
  if ~isfield(opts, 'max_particles') || isempty(opts.max_particles), opts.max_particles = 2e8; end

  kB = 1.380649e-23;
  hbar = 1.054571817e-34;

  [Nc, Vc] = local_cell_volumes_(mesh);
  Vdom = sum(Vc);
  [B, Nw] = size(spec.w_mid);

  [Tcell, Tmeta] = local_initial_temperature_field_(mesh, opts);
  [Tref_cell, Tref_meta] = local_reference_temperature_field_(mesh, opts, Tcell);
  [mode_weight, mode_sign, U_density_cell] = local_cell_mode_spectra_(spec, opts, Tcell, Tref_cell, hbar, kB);

  af = local_enhance_factor_(opts, Nc);
  mode_energy_abs = sum(mode_weight, 1).';
  cell_weight = mode_energy_abs .* Vc(:) .* af;
  U_total = sum(U_density_cell(:) .* Vc(:));

  Nexp_tot = sum(cell_weight) / opts.E_eff;
  Nsp_tot = floor(Nexp_tot) + (rand < (Nexp_tot - floor(Nexp_tot)));

  if Nsp_tot <= 0
      state = local_empty_state_();
      state.WE = opts.E_eff;
      state.Wp = opts.E_eff;
      state.enhance_factor = af;
      state.info = struct( ...
          'mode', opts.mode, ...
          'Tref', local_reference_temperature_(opts), ...
          'Tref_cell', Tref_cell, ...
          'reference_temperature_meta', Tref_meta, ...
          'T_init_cell', Tcell, ...
          'initial_temperature_meta', Tmeta, ...
          'U_total', U_total, ...
          'U_density_mean', U_total / max(Vdom, realmin), ...
          'Nexp_tot', Nexp_tot, ...
          'Nsp_tot', 0, ...
          'Nc', Nc, ...
          'Vdom', Vdom);
      return;
  end

  if Nsp_tot > opts.max_particles
      error('init_state_Energy: expected particles %.3g exceed max_particles %.3g.', Nsp_tot, opts.max_particles);
  end

  cdf_cell = cumsum(cell_weight) / sum(cell_weight);
  mode_cdf = zeros(B * Nw, Nc);
  for cid = 1:Nc
      if mode_energy_abs(cid) <= 0
          continue;
      end
      mode_cdf(:, cid) = cumsum(mode_weight(:, cid)) / mode_energy_abs(cid);
  end

  have_w_edges = isfield(spec, 'w_edges') && numel(spec.w_edges) == (Nw + 1);
  p(Nsp_tot, 1) = local_blank_particle_();

  for i = 1:Nsp_tot
      cid = local_sample_from_cdf_(cdf_cell);
      [x, y, z] = local_uniform_position_in_cell_(mesh, cid);

      idx = local_sample_from_cdf_(mode_cdf(:, cid));
      b = 1 + mod(idx - 1, B);
      m = floor((idx - 1) / B) + 1;

      if opts.use_bin_center_w || ~have_w_edges
          w = spec.w_mid(b, m);
      else
          w = spec.w_edges(m) + (spec.w_edges(m + 1) - spec.w_edges(m)) * rand();
      end

      [q, vabs] = local_q_vabs_from_w_table_(w, spec, b);
      dir = local_rand_unit_vec_();

      sgn = +1;
      E = opts.E_eff;
      if strcmpi(opts.mode, 'deviational')
          sgn = mode_sign(idx, cid);
          E = sgn * opts.E_eff;
      end

      p(i).id = i;
      p(i).par_id = i;
      p(i).cell = cid;
      p(i).x = x; p(i).y = y; p(i).z = z;
      p(i).b = b; p(i).m = m; p(i).w = w; p(i).q = q;
      p(i).vabs = vabs; p(i).v = vabs * dir;
      p(i).E = E; p(i).sgn = sgn;
      p(i).n_ph = E / (hbar * max(w, 1e-30));
      p(i).seed = randi(2^31 - 1);
      p(i).t_left = 0;
  end

  Nsp_cell = accumarray([p.cell].', 1, [Nc 1], @sum, 0);
  state = struct();
  state.p = p;
  state.WE = opts.E_eff;
  state.Wp = opts.E_eff;
  state.Nsp_cell = Nsp_cell;
  state.enhance_factor = af;
  state.info = struct( ...
      'mode', opts.mode, ...
      'Tref', local_reference_temperature_(opts), ...
      'Tref_cell', Tref_cell, ...
      'reference_temperature_meta', Tref_meta, ...
      'T_init_cell', Tcell, ...
      'initial_temperature_meta', Tmeta, ...
      'U_density_cell', U_density_cell, ...
      'U_density_mean', U_total / max(Vdom, realmin), ...
      'U_total', U_total, ...
      'Nexp_tot', Nexp_tot, ...
      'Nsp_tot', Nsp_tot, ...
      'Nc', Nc, ...
      'Vdom', Vdom);
end

function [Tcell, meta] = local_initial_temperature_field_(mesh, opts)
  default_T = NaN;
  if strcmpi(opts.mode, 'absolute')
      if isfield(opts, 'T_init') && isfinite(opts.T_init)
          default_T = opts.T_init;
      elseif isfield(opts, 'T0') && isfinite(opts.T0)
          default_T = opts.T0;
      end
  else
      if isfield(opts, 'T0') && isfinite(opts.T0)
          default_T = opts.T0;
      elseif isfield(opts, 'Tref') && isfinite(opts.Tref)
          default_T = opts.Tref;
      end
  end

  [Tcell, meta] = load_initial_temperature_field_(mesh, opts, default_T);
end

function [Tref_cell, meta] = local_reference_temperature_field_(mesh, opts, Tcell)
  default_Tref = NaN;
  if isfield(opts, 'Tref') && isfinite(opts.Tref)
      default_Tref = opts.Tref;
  elseif isfield(opts, 'T0') && isfinite(opts.T0)
      default_Tref = opts.T0;
  elseif ~isempty(Tcell)
      default_Tref = mean(Tcell);
  end

  [Tref_cell, meta] = load_reference_temperature_field_(mesh, opts, default_Tref);
end

function [mode_weight, mode_sign, U_density_cell] = local_cell_mode_spectra_(spec, opts, Tcell, Tref_cell, hbar, kB)
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
  U_density_cell = zeros(Nc, 1);

  for cid = 1:Nc
      n_cell = local_bose_(spec.w_mid, Tcell(cid), hbar, kB);
      if strcmpi(opts.mode, 'deviational')
          n_ref = local_bose_(spec.w_mid, Tref_cell(cid), hbar, kB);
          Wbm = pref .* (n_cell - n_ref);
      else
          Wbm = pref .* n_cell;
      end

      mode_weight(:, cid) = abs(Wbm(:));
      if strcmpi(opts.mode, 'deviational')
          sgn = sign(Wbm(:));
          sgn(sgn == 0) = +1;
          mode_sign(:, cid) = sgn;
      end
      U_density_cell(cid) = sum(Wbm(:));
  end
end

function nBE = local_bose_(w, T, hbar, kB)
  T = max(T, 1e-12);
  x = hbar .* w ./ (kB * T);
  nBE = 1 ./ max(exp(min(x, 700)) - 1, realmin);
end

function af = local_enhance_factor_(opts, Nc)
  af = opts.enhance_factor;
  if isscalar(af)
      af = repmat(af, Nc, 1);
      return;
  end

  af = af(:);
  if numel(af) ~= Nc
      error('init_state_Energy: enhance_factor must be scalar or Nc-by-1.');
  end
end

function Tref = local_reference_temperature_(opts)
  if isfield(opts, 'Tref') && isfinite(opts.Tref)
      Tref = opts.Tref;
  elseif isfield(opts, 'T0') && isfinite(opts.T0)
      Tref = opts.T0;
  else
      Tref = 300;
  end
end

function S = local_empty_state_()
  S = struct('p', [], 'WE', [], 'Wp', [], 'Nsp_cell', [], ...
             'E_target', [], 'E_current', [], 'enhance_factor', [], ...
             'info', struct());
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

  error('init_state_Energy: mesh does not provide usable cell volumes.');
end

function [x, y, z] = local_uniform_position_in_cell_(mesh, cid)
  epsl = 1e-12;
  if all(isfield(mesh, {'Nx', 'Ny', 'Nz', 'x_edges', 'y_edges', 'z_edges'}))
      [ix, iy, iz] = ind2sub([mesh.Nx, mesh.Ny, mesh.Nz], cid);
      X = mesh.x_edges(:); Y = mesh.y_edges(:); Z = mesh.z_edges(:);
      x = X(ix) + (X(ix + 1) - X(ix)) * rand();
      y = Y(iy) + (Y(iy + 1) - Y(iy)) * rand();
      z = Z(iz) + (Z(iz + 1) - Z(iz)) * rand();
      x = min(max(x, X(ix) + epsl), X(ix + 1) - epsl);
      y = min(max(y, Y(iy) + epsl), Y(iy + 1) - epsl);
      z = min(max(z, Z(iz) + epsl), Z(iz + 1) - epsl);
      return;
  end

  if isfield(mesh, 'boxes') && ~isempty(mesh.boxes)
      b = mesh.boxes(cid, :);
      x = b(1) + (b(2) - b(1)) * rand();
      y = b(3) + (b(4) - b(3)) * rand();
      z = b(5) + (b(6) - b(5)) * rand();
      x = min(max(x, b(1) + epsl), b(2) - epsl);
      y = min(max(y, b(3) + epsl), b(4) - epsl);
      z = min(max(z, b(5) + epsl), b(6) - epsl);
      return;
  end

  error('init_state_Energy: mesh does not provide usable cell boxes.');
end

function idx = local_sample_from_cdf_(cdf)
  r = rand();
  idx = find(cdf >= r, 1, 'first');
  if isempty(idx), idx = numel(cdf); end
end

function u = local_rand_unit_vec_()
  u1 = rand();
  u2 = rand();
  cz = 2 * u1 - 1;
  sz = sqrt(max(0, 1 - cz^2));
  phi = 2 * pi * u2;
  u = [sz * cos(phi), sz * sin(phi), cz];
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
