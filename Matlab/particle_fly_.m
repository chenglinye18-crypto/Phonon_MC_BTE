function [state, absorb] = particle_fly_(state, mesh, dt, opts, spec) %#ok<INUSD>
% particle_fly_ Advect particles and apply rule-driven face actions.
%
% Supported face actions:
%   PASS      pass through the face
%   REFLECT   specular reflection on that face
%   CATCH     remove the particle
%   GENERATE  split into one PASS branch and one REFLECT branch

  if nargin < 4 || isempty(opts), opts = struct(); end
  if ~isfield(opts, 'fly_mode') || isempty(opts.fly_mode), opts.fly_mode = 'cell'; end

  must = {'Nx', 'Ny', 'Nz', 'x_edges', 'y_edges', 'z_edges'};
  if ~all(isfield(mesh, must))
      error('particle_fly_: mesh must provide Nx/Ny/Nz and explicit edges.');
  end

  absorb = struct('heat_flux', local_empty_heat_flux_stats_(mesh));
  switch lower(opts.fly_mode)
    case {'cell', 'domain'}
      [state, absorb.heat_flux] = local_fly_cell_faces_(state, mesh, dt, opts);
    otherwise
      error('particle_fly_: unknown fly_mode "%s".', opts.fly_mode);
  end
end

function [state, heat_flux_stats] = local_fly_cell_faces_(state, mesh, dt, opts)
  p = state.p;
  heat_flux_stats = local_empty_heat_flux_stats_(mesh);
  if isempty(p), return; end

  Nx = mesh.Nx; Ny = mesh.Ny; Nz = mesh.Nz;
  X = mesh.x_edges(:); Y = mesh.y_edges(:); Z = mesh.z_edges(:);
  epsl = 1e-11;

  logcfg = local_getfield_or_(opts, 'log', struct());
  chunk_size = local_getfield_or_(logcfg, 'fly_chunk', 2e5);
  if ~isscalar(chunk_size) || chunk_size <= 0
      chunk_size = numel(p);
  end

  next_id = local_next_particle_id_(p);
  nBlocks = ceil(numel(p) / chunk_size);
  survivors_blocks = cell(nBlocks, 1);

  for bId = 1:nBlocks
      i1 = (bId - 1) * chunk_size + 1;
      i2 = min(bId * chunk_size, numel(p));
      block_p = p(i1:i2);
      if isempty(block_p)
          survivors_blocks{bId} = p([]);
          continue;
      end

      [block_p, next_id, block_flux_stats] = local_fly_block_( ...
          block_p, mesh, X, Y, Z, Nx, Ny, Nz, dt, epsl, next_id);
      survivors_blocks{bId} = block_p;
      heat_flux_stats = local_merge_heat_flux_stats_(heat_flux_stats, block_flux_stats);
  end

  nonempty = ~cellfun(@isempty, survivors_blocks);
  if any(nonempty)
      state.p = vertcat(survivors_blocks{nonempty});
  else
      state.p = p([]);
  end
end

function [block_p, next_id, heat_flux_stats] = local_fly_block_(block_p, mesh, X, Y, Z, Nx, Ny, Nz, dt, epsl, next_id)
  nc = numel(block_p);
  heat_flux_stats = local_empty_heat_flux_stats_(mesh);
  x = [block_p.x].';
  y = [block_p.y].';
  z = [block_p.z].';
  v = vertcat(block_p.v);
  vx = v(:, 1);
  vy = v(:, 2);
  vz = v(:, 3);
  vabs = [block_p.vabs].';
  cid = double([block_p.cell].');
  alive = (cid >= 1) & (cid <= Nx * Ny * Nz) & (vabs > 0);
  t_rem = dt * ones(nc, 1);

  cid_safe = max(min(cid, Nx * Ny * Nz), 1);
  [ix, iy, iz] = ind2sub([Nx, Ny, Nz], cid_safe);

  if any(alive)
      alive_idx = find(alive);
      x(alive_idx) = min(max(x(alive_idx), X(ix(alive_idx)) + epsl), X(ix(alive_idx) + 1) - epsl);
      y(alive_idx) = min(max(y(alive_idx), Y(iy(alive_idx)) + epsl), Y(iy(alive_idx) + 1) - epsl);
      z(alive_idx) = min(max(z(alive_idx), Z(iz(alive_idx)) + epsl), Z(iz(alive_idx) + 1) - epsl);
  end

  while true
      act = find(alive & (t_rem > 0));
      if isempty(act), break; end

      INF = inf(numel(act), 1);
      tx = INF; ty = INF; tz = INF;

      xa = x(act); ya = y(act); za = z(act);
      vxa = vx(act); vya = vy(act); vza = vz(act);
      ix_a = ix(act); iy_a = iy(act); iz_a = iz(act);

      xL = X(ix_a); xR = X(ix_a + 1);
      yB = Y(iy_a); yT = Y(iy_a + 1);
      zD = Z(iz_a); zU = Z(iz_a + 1);

      pos = vxa > 0; if any(pos), tx(pos) = (xR(pos) - xa(pos)) ./ vxa(pos); end
      neg = vxa < 0; if any(neg), tx(neg) = (xL(neg) - xa(neg)) ./ vxa(neg); end
      pos = vya > 0; if any(pos), ty(pos) = (yT(pos) - ya(pos)) ./ vya(pos); end
      neg = vya < 0; if any(neg), ty(neg) = (yB(neg) - ya(neg)) ./ vya(neg); end
      pos = vza > 0; if any(pos), tz(pos) = (zU(pos) - za(pos)) ./ vza(pos); end
      neg = vza < 0; if any(neg), tz(neg) = (zD(neg) - za(neg)) ./ vza(neg); end

      tx = max(tx, 0);
      ty = max(ty, 0);
      tz = max(tz, 0);

      tcell = tx;
      axis_ix = ones(numel(act), 1, 'uint8');
      m = ty < tcell; tcell(m) = ty(m); axis_ix(m) = 2;
      m = tz < tcell; tcell(m) = tz(m); axis_ix(m) = 3;

      tf = min(tcell, t_rem(act));
      x(act) = x(act) + vx(act) .* tf;
      y(act) = y(act) + vy(act) .* tf;
      z(act) = z(act) + vz(act) .* tf;
      t_rem(act) = t_rem(act) - tf;

      hit = isfinite(tcell) & (abs(tf - tcell) <= local_rel_tol_(tcell));
      hit_ids = find(hit).';
      for jj = hit_ids
          k = act(jj);
          switch axis_ix(jj)
            case 1
              if vx(k) > 0
                  normal = '+X';
              else
                  normal = '-X';
              end
            case 2
              if vy(k) > 0
                  normal = '+Y';
              else
                  normal = '-Y';
              end
            otherwise
              if vz(k) > 0
                  normal = '+Z';
              else
                  normal = '-Z';
              end
          end

          pt = [x(k), y(k), z(k)];
          action = local_face_action_(mesh, normal, pt);
          [block_p, x, y, z, vx, vy, vz, vabs, cid, ix, iy, iz, alive, t_rem, next_id, heat_flux_stats] = ...
              local_apply_action_(block_p, x, y, z, vx, vy, vz, vabs, cid, ix, iy, iz, alive, t_rem, ...
                                  k, normal, action, X, Y, Z, Nx, Ny, Nz, epsl, next_id, mesh, pt, heat_flux_stats);
      end

      alive_idx = find(alive);
      if ~isempty(alive_idx)
          x(alive_idx) = min(max(x(alive_idx), X(ix(alive_idx)) + epsl), X(ix(alive_idx) + 1) - epsl);
          y(alive_idx) = min(max(y(alive_idx), Y(iy(alive_idx)) + epsl), Y(iy(alive_idx) + 1) - epsl);
          z(alive_idx) = min(max(z(alive_idx), Z(iz(alive_idx)) + epsl), Z(iz(alive_idx) + 1) - epsl);
      end
  end

  keep = alive & (cid > 0);
  keep_idx = find(keep);
  block_p = block_p(keep_idx);
  for i = 1:numel(keep_idx)
      src = keep_idx(i);
      block_p(i).x = x(src);
      block_p(i).y = y(src);
      block_p(i).z = z(src);
      block_p(i).v = [vx(src), vy(src), vz(src)];
      block_p(i).vabs = vabs(src);
      block_p(i).cell = int32(cid(src));
      block_p(i).t_left = 0;
  end
end

function [block_p, x, y, z, vx, vy, vz, vabs, cid, ix, iy, iz, alive, t_rem, next_id, heat_flux_stats] = ...
    local_apply_action_(block_p, x, y, z, vx, vy, vz, vabs, cid, ix, iy, iz, alive, t_rem, ...
                        k, normal, action, X, Y, Z, Nx, Ny, Nz, epsl, next_id, mesh, pt, heat_flux_stats)

  packet_E = local_particle_energy_(block_p(k));

  switch action
    case 'pass'
      heat_flux_stats = local_tally_heat_flux_crossing_(heat_flux_stats, mesh, pt, normal, packet_E);
      [has_neighbor, x_new, y_new, z_new, ix_new, iy_new, iz_new] = ...
          local_pass_state_(normal, x(k), y(k), z(k), ix(k), iy(k), iz(k), X, Y, Z, Nx, Ny, Nz, epsl);
      if has_neighbor
          x(k) = x_new; y(k) = y_new; z(k) = z_new;
          ix(k) = ix_new; iy(k) = iy_new; iz(k) = iz_new;
          cid(k) = sub2ind([Nx, Ny, Nz], ix_new, iy_new, iz_new);
      else
          alive(k) = false;
          cid(k) = -1;
      end

    case 'reflect'
      [x(k), y(k), z(k), vx(k), vy(k), vz(k)] = ...
          local_reflect_state_(normal, x(k), y(k), z(k), vx(k), vy(k), vz(k), ix(k), iy(k), iz(k), X, Y, Z, epsl);

    case 'catch'
      alive(k) = false;
      cid(k) = -1;

    case 'periodic'
      [x(k), y(k), z(k), ix(k), iy(k), iz(k)] = ...
          local_periodic_state_(normal, x(k), y(k), z(k), ix(k), iy(k), iz(k), X, Y, Z, Nx, Ny, Nz, epsl);
      cid(k) = sub2ind([Nx, Ny, Nz], ix(k), iy(k), iz(k));

    case 'generate'
      [has_neighbor, x_pass, y_pass, z_pass, ix_pass, iy_pass, iz_pass] = ...
          local_pass_state_(normal, x(k), y(k), z(k), ix(k), iy(k), iz(k), X, Y, Z, Nx, Ny, Nz, epsl);

      if has_neighbor
          heat_flux_stats = local_tally_heat_flux_crossing_(heat_flux_stats, mesh, pt, normal, packet_E);
          next_id = next_id + 1;
          child = block_p(k);
          child.id = next_id;
          child.par_id = next_id;
          child.seed = randi(2^31 - 1);
          child.x = x_pass;
          child.y = y_pass;
          child.z = z_pass;
          child.v = [vx(k), vy(k), vz(k)];
          child.vabs = vabs(k);
          child.cell = int32(sub2ind([Nx, Ny, Nz], ix_pass, iy_pass, iz_pass));
          child.t_left = t_rem(k);

          block_p(end + 1, 1) = child; %#ok<AGROW>
          x(end + 1, 1) = x_pass; %#ok<AGROW>
          y(end + 1, 1) = y_pass; %#ok<AGROW>
          z(end + 1, 1) = z_pass; %#ok<AGROW>
          vx(end + 1, 1) = vx(k); %#ok<AGROW>
          vy(end + 1, 1) = vy(k); %#ok<AGROW>
          vz(end + 1, 1) = vz(k); %#ok<AGROW>
          vabs(end + 1, 1) = vabs(k); %#ok<AGROW>
          ix(end + 1, 1) = ix_pass; %#ok<AGROW>
          iy(end + 1, 1) = iy_pass; %#ok<AGROW>
          iz(end + 1, 1) = iz_pass; %#ok<AGROW>
          cid(end + 1, 1) = sub2ind([Nx, Ny, Nz], ix_pass, iy_pass, iz_pass); %#ok<AGROW>
          alive(end + 1, 1) = true; %#ok<AGROW>
          t_rem(end + 1, 1) = t_rem(k); %#ok<AGROW>
      end

      [x(k), y(k), z(k), vx(k), vy(k), vz(k)] = ...
          local_reflect_state_(normal, x(k), y(k), z(k), vx(k), vy(k), vz(k), ix(k), iy(k), iz(k), X, Y, Z, epsl);

    otherwise
      error('particle_fly_: unsupported face action "%s".', action);
  end
end

function [has_neighbor, x_new, y_new, z_new, ix_new, iy_new, iz_new] = ...
    local_pass_state_(normal, x, y, z, ix, iy, iz, X, Y, Z, Nx, Ny, Nz, epsl)

  has_neighbor = true;
  x_new = x; y_new = y; z_new = z;
  ix_new = ix; iy_new = iy; iz_new = iz;

  switch upper(normal)
    case '+X'
      if ix >= Nx
          has_neighbor = false;
      else
          ix_new = ix + 1;
          x_new = X(ix_new) + epsl;
      end
    case '-X'
      if ix <= 1
          has_neighbor = false;
      else
          ix_new = ix - 1;
          x_new = X(ix_new + 1) - epsl;
      end
    case '+Y'
      if iy >= Ny
          has_neighbor = false;
      else
          iy_new = iy + 1;
          y_new = Y(iy_new) + epsl;
      end
    case '-Y'
      if iy <= 1
          has_neighbor = false;
      else
          iy_new = iy - 1;
          y_new = Y(iy_new + 1) - epsl;
      end
    case '+Z'
      if iz >= Nz
          has_neighbor = false;
      else
          iz_new = iz + 1;
          z_new = Z(iz_new) + epsl;
      end
    case '-Z'
      if iz <= 1
          has_neighbor = false;
      else
          iz_new = iz - 1;
          z_new = Z(iz_new + 1) - epsl;
      end
    otherwise
      error('particle_fly_: invalid normal %s.', normal);
  end
end

function [x, y, z, vx, vy, vz] = local_reflect_state_(normal, x, y, z, vx, vy, vz, ix, iy, iz, X, Y, Z, epsl)
  switch upper(normal)
    case '+X'
      vx = -vx;
      x = X(ix + 1) - epsl;
    case '-X'
      vx = -vx;
      x = X(ix) + epsl;
    case '+Y'
      vy = -vy;
      y = Y(iy + 1) - epsl;
    case '-Y'
      vy = -vy;
      y = Y(iy) + epsl;
    case '+Z'
      vz = -vz;
      z = Z(iz + 1) - epsl;
    case '-Z'
      vz = -vz;
      z = Z(iz) + epsl;
    otherwise
      error('particle_fly_: invalid normal %s.', normal);
  end
end

function [x, y, z, ix, iy, iz] = local_periodic_state_(normal, x, y, z, ix, iy, iz, X, Y, Z, Nx, Ny, Nz, epsl)
  switch upper(normal)
    case '+X'
      ix = 1;
      x = X(1) + epsl;
    case '-X'
      ix = Nx;
      x = X(end) - epsl;
    case '+Y'
      iy = 1;
      y = Y(1) + epsl;
    case '-Y'
      iy = Ny;
      y = Y(end) - epsl;
    case '+Z'
      iz = 1;
      z = Z(1) + epsl;
    case '-Z'
      iz = Nz;
      z = Z(end) - epsl;
    otherwise
      error('particle_fly_: invalid normal %s.', normal);
  end
end

function action = local_face_action_(mesh, normal, pt)
  action = 'pass';
  if isfield(mesh, 'face_rules') && isfield(mesh.face_rules, 'by_normal')
      key = local_normal_key_(normal);
      if isfield(mesh.face_rules.by_normal, key)
          rules = mesh.face_rules.by_normal.(key);
          tol = 1e-12 * max(1, max(abs(pt)));
          for i = 1:numel(rules)
              if local_point_hits_rule_(pt, rules(i), tol)
                  action = local_normalize_action_(rules(i).mode);
                  return;
              end
          end
      end
  end
end

function tf = local_point_hits_rule_(pt, rule, tol)
  b = rule.bounds;
  switch rule.axis
    case 'x'
      tf = abs(pt(1) - rule.coord) <= tol && ...
           pt(2) >= b(3) - tol && pt(2) <= b(4) + tol && ...
           pt(3) >= b(5) - tol && pt(3) <= b(6) + tol;
    case 'y'
      tf = abs(pt(2) - rule.coord) <= tol && ...
           pt(1) >= b(1) - tol && pt(1) <= b(2) + tol && ...
           pt(3) >= b(5) - tol && pt(3) <= b(6) + tol;
    case 'z'
      tf = abs(pt(3) - rule.coord) <= tol && ...
           pt(1) >= b(1) - tol && pt(1) <= b(2) + tol && ...
           pt(2) >= b(3) - tol && pt(2) <= b(4) + tol;
    otherwise
      tf = false;
  end
end

function action = local_normalize_action_(mode_name)
  switch lower(mode_name)
    case {'pass', 'open'}
      action = 'pass';
    case {'reflect', 'adiabatic'}
      action = 'reflect';
    case {'catch', 'absorb'}
      action = 'catch';
    case 'generate'
      action = 'generate';
    case 'periodic'
      action = 'periodic';
    otherwise
      action = 'pass';
  end
end

function key = local_normal_key_(normal_name)
  switch upper(normal_name)
    case '+X'
      key = 'xp';
    case '-X'
      key = 'xn';
    case '+Y'
      key = 'yp';
    case '-Y'
      key = 'yn';
    case '+Z'
      key = 'zp';
    case '-Z'
      key = 'zn';
    otherwise
      error('particle_fly_: invalid normal %s.', normal_name);
  end
end

function tol = local_rel_tol_(t)
  tol = max(1e-15 .* t, 1e-18);
end

function next_id = local_next_particle_id_(p)
  if isempty(p)
      next_id = 0;
  else
      next_id = max([p.id]);
  end
end

function v = local_getfield_or_(s, name, default_v)
  if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
      v = s.(name);
  else
      v = default_v;
  end
end

function stats = local_empty_heat_flux_stats_(mesh)
  nmon = 0;
  if isfield(mesh, 'heat_flux_monitors') && ~isempty(mesh.heat_flux_monitors)
      nmon = numel(mesh.heat_flux_monitors);
  end
  stats = struct('net_energy', zeros(nmon, 1), ...
                 'forward_energy', zeros(nmon, 1), ...
                 'backward_energy', zeros(nmon, 1), ...
                 'crossings_pos', zeros(nmon, 1), ...
                 'crossings_neg', zeros(nmon, 1));
end

function stats = local_merge_heat_flux_stats_(stats, add_stats)
  if isempty(add_stats)
      return;
  end
  stats.net_energy = stats.net_energy + add_stats.net_energy;
  stats.forward_energy = stats.forward_energy + add_stats.forward_energy;
  stats.backward_energy = stats.backward_energy + add_stats.backward_energy;
  stats.crossings_pos = stats.crossings_pos + add_stats.crossings_pos;
  stats.crossings_neg = stats.crossings_neg + add_stats.crossings_neg;
end

function stats = local_tally_heat_flux_crossing_(stats, mesh, pt, normal, packet_E)
  if ~isfield(mesh, 'heat_flux_monitors') || isempty(mesh.heat_flux_monitors)
      return;
  end

  for i = 1:numel(mesh.heat_flux_monitors)
      mon = mesh.heat_flux_monitors(i);
      if ~local_point_hits_monitor_(pt, mon)
          continue;
      end

      dir_sign = local_crossing_sign_(normal, mon.effective_normal);
      if dir_sign == 0
          continue;
      end

      stats.net_energy(i) = stats.net_energy(i) + dir_sign * packet_E;
      if dir_sign > 0
          stats.forward_energy(i) = stats.forward_energy(i) + packet_E;
          stats.crossings_pos(i) = stats.crossings_pos(i) + 1;
      else
          stats.backward_energy(i) = stats.backward_energy(i) + packet_E;
          stats.crossings_neg(i) = stats.crossings_neg(i) + 1;
      end
  end
end

function tf = local_point_hits_monitor_(pt, mon)
  b = mon.bounds;
  tol = 1e-12 * max(1, max(abs([pt(:); b(:)])));
  switch mon.axis
    case 'x'
      tf = abs(pt(1) - mon.coord) <= tol && ...
           pt(2) >= b(3) - tol && pt(2) <= b(4) + tol && ...
           pt(3) >= b(5) - tol && pt(3) <= b(6) + tol;
    case 'y'
      tf = abs(pt(2) - mon.coord) <= tol && ...
           pt(1) >= b(1) - tol && pt(1) <= b(2) + tol && ...
           pt(3) >= b(5) - tol && pt(3) <= b(6) + tol;
    case 'z'
      tf = abs(pt(3) - mon.coord) <= tol && ...
           pt(1) >= b(1) - tol && pt(1) <= b(2) + tol && ...
           pt(2) >= b(3) - tol && pt(2) <= b(4) + tol;
    otherwise
      tf = false;
  end
end

function dir_sign = local_crossing_sign_(face_normal, monitor_normal)
  if strcmpi(face_normal, monitor_normal)
      dir_sign = +1;
  elseif strcmpi(face_normal, local_opposite_normal_(monitor_normal))
      dir_sign = -1;
  else
      dir_sign = 0;
  end
end

function normal_name = local_opposite_normal_(normal_name)
  switch upper(normal_name)
    case '+X'
      normal_name = '-X';
    case '-X'
      normal_name = '+X';
    case '+Y'
      normal_name = '-Y';
    case '-Y'
      normal_name = '+Y';
    case '+Z'
      normal_name = '-Z';
    case '-Z'
      normal_name = '+Z';
    otherwise
      error('particle_fly_: invalid normal %s.', normal_name);
  end
end

function E = local_particle_energy_(p)
  if isfield(p, 'E') && isfinite(p.E)
      E = p.E;
  elseif isfield(p, 'sgn') && isfinite(p.sgn)
      E = sign(p.sgn);
      if E == 0
          E = 1;
      end
  else
      E = 0;
  end
end
