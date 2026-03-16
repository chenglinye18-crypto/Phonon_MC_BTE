function mesh = init_mesh_from_geom_(cs)
% init_mesh_from_geom_ Build the runtime mesh from the parsed case geometry.
%
% Supported inputs:
%   1) Explicit edges
%      cs.mesh.x_edges / y_edges / z_edges
%   2) Uniform counts
%      cs.geom.L = [Lx Ly Lz]
%      cs.mesh.Nx / Ny / Nz
%
% Besides geometry, this function also attaches:
%   - region-to-cell material labels
%   - face-rule lookup tables used by particle_fly_
%   - reservoir-to-cell mappings used by periodic reservoir refresh

  [X, Y, Z, Lx, Ly, Lz] = local_resolve_edges_(cs);

  Nx = numel(X) - 1;
  Ny = numel(Y) - 1;
  Nz = numel(Z) - 1;
  Nc = Nx * Ny * Nz;

  dx = diff(X);
  dy = diff(Y);
  dz = diff(Z);
  hmin = min([dx(:); dy(:); dz(:)]);

  xc = 0.5 * (X(1:end-1) + X(2:end));
  yc = 0.5 * (Y(1:end-1) + Y(2:end));
  zc = 0.5 * (Z(1:end-1) + Z(2:end));
  [Xc, Yc, Zc] = ndgrid(xc, yc, zc);

  [I, J, K] = ndgrid(1:Nx, 1:Ny, 1:Nz);
  I = I(:);
  J = J(:);
  K = K(:);

  xmin = X(I);
  xmax = X(I + 1);
  ymin = Y(J);
  ymax = Y(J + 1);
  zmin = Z(K);
  zmax = Z(K + 1);

  centers = [xc(I), yc(J), zc(K)];
  boxes = [xmin, xmax, ymin, ymax, zmin, zmax];
  vol = (xmax - xmin) .* (ymax - ymin) .* (zmax - zmin);

  to_id = @(i,j,k) sub2ind([Nx, Ny, Nz], i, j, k);
  from_id = @(id) local_from_id_(id, [Nx, Ny, Nz]);
  point2id = @(x,y,z) local_point2id_regular_(x, y, z, X, Y, Z, Nx, Ny, Nz, to_id);

  mesh = struct();
  mesh.L = [Lx, Ly, Lz];
  mesh.Lx = Lx;
  mesh.Ly = Ly;
  mesh.Lz = Lz;
  mesh.origin = [X(1), Y(1), Z(1)];
  mesh.Nx = Nx;
  mesh.Ny = Ny;
  mesh.Nz = Nz;
  mesh.Nc = Nc;
  mesh.dx = dx;
  mesh.dy = dy;
  mesh.dz = dz;
  mesh.dx_min = min(dx);
  mesh.dy_min = min(dy);
  mesh.dz_min = min(dz);
  mesh.hmin = hmin;

  mesh.centers = centers;
  mesh.vol = vol;
  mesh.cell_vol = vol;
  mesh.boxes = boxes;

  mesh.to_id = to_id;
  mesh.from_id = from_id;
  mesh.point2id = point2id;

  mesh.domain_box = [X(1), X(end), Y(1), Y(end), Z(1), Z(end)];
  mesh.x_edges = X;
  mesh.y_edges = Y;
  mesh.z_edges = Z;
  mesh.xc = xc;
  mesh.yc = yc;
  mesh.zc = zc;
  mesh.Xc = Xc;
  mesh.Yc = Yc;
  mesh.Zc = Zc;

  mesh.Ax = Ly * Lz;
  mesh.Ay = Lx * Lz;
  mesh.Az = Lx * Ly;
  mesh.V = sum(vol);

  mesh = local_copy_if_present_(mesh, cs, 'regions');
  mesh = local_copy_if_present_(mesh, cs, 'materials');
  mesh = local_copy_if_present_(mesh, cs, 'layout');
  mesh = local_copy_if_present_(mesh, cs, 'grid');
  mesh = local_attach_cell_materials_(mesh);
  mesh = local_attach_reservoirs_(mesh);

  if isfield(mesh, 'layout') && ~isempty(mesh.layout)
      mesh = build_layout_behavior_(mesh, mesh.layout);
  else
      mesh.boundary = struct('by_face', struct());
      mesh.face_rules = struct('by_normal', struct( ...
          'xp', struct([]), 'xn', struct([]), ...
          'yp', struct([]), 'yn', struct([]), ...
          'zp', struct([]), 'zn', struct([])), ...
          'all', struct([]));
  end
end

function [X, Y, Z, Lx, Ly, Lz] = local_resolve_edges_(cs)
  has_explicit = isfield(cs, 'mesh') && all(isfield(cs.mesh, {'x_edges', 'y_edges', 'z_edges'}));

  if has_explicit
      X = cs.mesh.x_edges(:);
      Y = cs.mesh.y_edges(:);
      Z = cs.mesh.z_edges(:);
      local_validate_edges_(X, 'x');
      local_validate_edges_(Y, 'y');
      local_validate_edges_(Z, 'z');

      Lx = X(end) - X(1);
      Ly = Y(end) - Y(1);
      Lz = Z(end) - Z(1);

      if isfield(cs, 'geom') && isfield(cs.geom, 'L') && numel(cs.geom.L) == 3
          gL = cs.geom.L(:).';
          tol = 1e-12 * max(1, max(abs([gL, Lx, Ly, Lz])));
          if any(abs(gL - [Lx, Ly, Lz]) > tol)
              error('init_mesh_from_geom_: geom.L disagrees with explicit mesh edges.');
          end
      end
      return;
  end

  assert(isfield(cs, 'geom') && isfield(cs.geom, 'L') && numel(cs.geom.L) == 3, ...
      'init_mesh_from_geom_: geom.L = [Lx Ly Lz] must be provided.');
  assert(isfield(cs, 'mesh') && all(isfield(cs.mesh, {'Nx', 'Ny', 'Nz'})), ...
      'init_mesh_from_geom_: either explicit edges or mesh.Nx/Ny/Nz must be provided.');

  origin = [0, 0, 0];
  if isfield(cs.geom, 'origin') && numel(cs.geom.origin) == 3
      origin = cs.geom.origin(:).';
  end

  Lx = cs.geom.L(1);
  Ly = cs.geom.L(2);
  Lz = cs.geom.L(3);
  assert(all([Lx, Ly, Lz] > 0), 'init_mesh_from_geom_: geometry lengths must be positive.');

  Nx = cs.mesh.Nx;
  Ny = cs.mesh.Ny;
  Nz = cs.mesh.Nz;
  assert(all([Nx, Ny, Nz] >= 1) && all(mod([Nx, Ny, Nz], 1) == 0), ...
      'init_mesh_from_geom_: mesh.Nx/Ny/Nz must be positive integers.');

  X = linspace(origin(1), origin(1) + Lx, Nx + 1).';
  Y = linspace(origin(2), origin(2) + Ly, Ny + 1).';
  Z = linspace(origin(3), origin(3) + Lz, Nz + 1).';
end

function local_validate_edges_(edges, tag)
  assert(isvector(edges) && numel(edges) >= 2, ...
      'init_mesh_from_geom_: mesh.%s_edges must contain at least two points.', tag);
  assert(all(isfinite(edges)), ...
      'init_mesh_from_geom_: mesh.%s_edges contains non-finite values.', tag);
  assert(all(diff(edges) > 0), ...
      'init_mesh_from_geom_: mesh.%s_edges must be strictly increasing.', tag);
end

function ijk = local_from_id_(id, sz)
  [i, j, k] = ind2sub(sz, id);
  ijk = [i, j, k];
end

function id = local_point2id_regular_(x, y, z, X, Y, Z, Nx, Ny, Nz, to_id)
  x = min(max(x, X(1)), X(end));
  y = min(max(y, Y(1)), Y(end));
  z = min(max(z, Z(1)), Z(end));

  ii = local_axis_to_index_(x, X, Nx);
  jj = local_axis_to_index_(y, Y, Ny);
  kk = local_axis_to_index_(z, Z, Nz);
  id = to_id(ii, jj, kk);
end

function idx = local_axis_to_index_(x, edges, N)
  idx = find(edges <= x, 1, 'last');
  if isempty(idx)
      idx = 1;
  elseif idx >= numel(edges)
      idx = N;
  end
  idx = min(max(idx, 1), N);
end

function mesh = local_copy_if_present_(mesh, cs, field_name)
  if isfield(cs, field_name)
      mesh.(field_name) = cs.(field_name);
  end
end

function mesh = local_attach_cell_materials_(mesh)
  if ~isfield(mesh, 'regions') || isempty(mesh.regions)
      return;
  end

  Nc = mesh.Nc;
  centers = mesh.centers;
  cell_region_index = zeros(Nc, 1, 'int32');

  for ir = 1:numel(mesh.regions)
      bounds = mesh.regions(ir).bounds;
      in_region = centers(:, 1) >= bounds(1) & centers(:, 1) <= bounds(2) & ...
                  centers(:, 2) >= bounds(3) & centers(:, 2) <= bounds(4) & ...
                  centers(:, 3) >= bounds(5) & centers(:, 3) <= bounds(6);
      cell_region_index(in_region) = int32(ir);
  end

  cell_material_name = repmat({''}, Nc, 1);
  for cid = 1:Nc
      ir = cell_region_index(cid);
      if ir >= 1
          cell_material_name{cid} = mesh.regions(ir).material;
      end
  end

  assigned = cell_material_name(~cellfun(@isempty, cell_material_name));
  if isempty(assigned)
      mesh.cell_region_index = cell_region_index;
      mesh.cell_material_name = cell_material_name;
      mesh.cell_material_index = zeros(Nc, 1, 'int32');
      mesh.material_keys = {};
      return;
  end

  material_keys = cellstr(unique(upper(string(assigned)), 'stable'));
  cell_material_index = zeros(Nc, 1, 'int32');
  for im = 1:numel(material_keys)
      cell_material_index(strcmpi(cell_material_name, material_keys{im})) = int32(im);
  end

  mesh.cell_region_index = cell_region_index;
  mesh.cell_material_name = cell_material_name;
  mesh.cell_material_index = cell_material_index;
  mesh.material_keys = material_keys(:).';
end

function mesh = local_attach_reservoirs_(mesh)
  if ~isfield(mesh, 'layout') || ~isstruct(mesh.layout) || ...
          ~isfield(mesh.layout, 'reservoirs') || isempty(mesh.layout.reservoirs)
      mesh.reservoirs = repmat(local_empty_mesh_reservoir_(), 0, 1);
      mesh.reservoir_cell_mask = false(mesh.Nc, 1);
      return;
  end

  reservoirs = repmat(local_empty_mesh_reservoir_(), numel(mesh.layout.reservoirs), 1);
  cell_mask = false(mesh.Nc, 1);
  ctr = mesh.centers;
  tol = 1e-12 * max(1, max(abs(ctr(:))));

  for ir = 1:numel(mesh.layout.reservoirs)
      src = mesh.layout.reservoirs(ir);
      b = src.bounds;
      in_res = ctr(:, 1) >= b(1) - tol & ctr(:, 1) <= b(2) + tol & ...
               ctr(:, 2) >= b(3) - tol & ctr(:, 2) <= b(4) + tol & ...
               ctr(:, 3) >= b(5) - tol & ctr(:, 3) <= b(6) + tol;
      cells = find(in_res);

      reservoirs(ir).id = src.id;
      reservoirs(ir).name = src.name;
      reservoirs(ir).bounds_input = src.bounds_input;
      reservoirs(ir).bounds = src.bounds;
      reservoirs(ir).cell_ids = cells(:);
      reservoirs(ir).raw = src.raw;
      cell_mask(cells) = true;
  end

  mesh.reservoirs = reservoirs;
  mesh.reservoir_cell_mask = cell_mask;
end

function res = local_empty_mesh_reservoir_()
  res = struct('id', int32(0), ...
               'name', '', ...
               'bounds_input', zeros(1, 6), ...
               'bounds', zeros(1, 6), ...
               'cell_ids', zeros(0, 1), ...
               'raw', '');
end
