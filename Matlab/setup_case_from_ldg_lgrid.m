function cs = setup_case_from_ldg_lgrid(varargin)
% setup_case_from_ldg_lgrid Build the full simulation case from ldg/lgrid text files.
%
% Supported ldg entries:
%   - variable definitions such as $Ly$ 0.58
%   - region x0 x1 y0 y1 z0 z1 MATERIAL
%   - planerule / lanerule x0 x1 y0 y1 z0 z1 NORMAL MODE
%   - RESERVOIR x0 x1 y0 y1 z0 z1
%
% Output:
%   cs.geom   : domain size in SI units
%   cs.mesh   : explicit edge-based mesh
%   cs.layout : parsed regions, face rules, and reservoirs
%   cs.grid   : raw lgrid-derived edge metadata

  p = inputParser;
  addParameter(p, 'LdgFile', fullfile('input', 'ldg.txt'));
  addParameter(p, 'LgridFile', fullfile('input', 'lgrid.txt'));
  addParameter(p, 'LengthScale', 1e-6);
  addParameter(p, 'InputLengthUnit', 'um');
  addParameter(p, 'Verbose', true);
  parse(p, varargin{:});

  cfg = p.Results;

  layout = local_parse_ldg_(cfg.LdgFile, cfg.LengthScale);
  grid = local_parse_lgrid_(cfg.LgridFile, cfg.LengthScale);
  local_validate_layout_vs_grid_(layout, grid);
  layout = local_build_layout_behavior_(layout);
  geom = local_build_geom_(grid);

  mesh = struct();
  mesh.Nx = grid.Nx;
  mesh.Ny = grid.Ny;
  mesh.Nz = grid.Nz;
  mesh.x_edges = grid.x_edges;
  mesh.y_edges = grid.y_edges;
  mesh.z_edges = grid.z_edges;

  units = struct('length', 'm', 'input_length', cfg.InputLengthUnit, 'temp', 'K');

  cs = struct();
  cs.units = units;
  cs.geom = geom;
  cs.mesh = mesh;
  cs.regions = layout.regions;
  cs.materials = layout.materials;
  cs.layout = layout;
  cs.grid = grid;

  if cfg.Verbose
      local_print_case_summary_(cs, cfg.LengthScale, geom);
  end
end

function layout = local_parse_ldg_(filepath, length_scale)
% local_parse_ldg_ Parse layout, face-rule, and reservoir definitions from ldg.
  lines = local_read_clean_lines_(filepath);

  vars_input = struct();
  vars_si = struct();
  regions = repmat(local_empty_region_(), 0, 1);
  rules = repmat(local_empty_rule_(), 0, 1);
  reservoirs = repmat(local_empty_reservoir_(), 0, 1);

  for i = 1:numel(lines)
      line = lines{i};
      tokens = regexp(line, '\s+', 'split');
      head = lower(tokens{1});

      if startsWith(tokens{1}, '$') && endsWith(tokens{1}, '$')
          var_name = regexprep(tokens{1}, '^\$|\$$', '');
          expr = strtrim(strjoin(tokens(2:end), ' '));
          value_input = local_eval_expr_(expr, vars_input);
          vars_input.(var_name) = value_input;
          vars_si.(var_name) = value_input * length_scale;
          continue;
      end

      switch head
        case 'region'
          if numel(tokens) < 8
              error('setup_case_from_ldg_lgrid: invalid region line: %s', line);
          end
          bounds_input = zeros(1, 6);
          for k = 1:6
              bounds_input(k) = local_eval_expr_(tokens{1 + k}, vars_input);
          end
          reg = local_empty_region_();
          reg.bounds_input = bounds_input;
          reg.bounds = bounds_input * length_scale;
          reg.material = tokens{8};
          reg.raw = line;
          regions(end + 1, 1) = reg; %#ok<AGROW>

        case {'planerule', 'lanerule'}
          if numel(tokens) < 9
              error('setup_case_from_ldg_lgrid: invalid plane rule line: %s', line);
          end
          bounds_input = zeros(1, 6);
          for k = 1:6
              bounds_input(k) = local_eval_expr_(tokens{1 + k}, vars_input);
          end

          rule = local_empty_rule_();
          rule.kind = head;
          rule.bounds_input = bounds_input;
          rule.bounds = bounds_input * length_scale;
          rule.normal = upper(tokens{8});
          rule.mode = upper(tokens{9});
          rule.raw = line;
          rules(end + 1, 1) = local_finalize_rule_(rule, vars_si); %#ok<AGROW>

        case 'reservoir'
          if numel(tokens) < 7
              error('setup_case_from_ldg_lgrid: invalid reservoir line: %s', line);
          end
          bounds_input = zeros(1, 6);
          for k = 1:6
              bounds_input(k) = local_eval_expr_(tokens{1 + k}, vars_input);
          end

          res = local_empty_reservoir_();
          res.id = int32(numel(reservoirs) + 1);
          res.name = sprintf('reservoir_%d', numel(reservoirs) + 1);
          res.bounds_input = bounds_input;
          res.bounds = bounds_input * length_scale;
          res.raw = line;
          reservoirs(end + 1, 1) = res; %#ok<AGROW>

        otherwise
          error('setup_case_from_ldg_lgrid: unsupported ldg entry: %s', line);
      end
  end

  materials = unique(string({regions.material}), 'stable');

  layout = struct();
  layout.source = filepath;
  layout.variables_input = vars_input;
  layout.variables_si = vars_si;
  layout.regions = regions;
  layout.materials = cellstr(materials);
  layout.rules = rules;
  layout.reservoirs = reservoirs;
end

function grid = local_parse_lgrid_(filepath, length_scale)
% local_parse_lgrid_ Expand compact axis anchor syntax into explicit mesh edges.
  lines = local_read_clean_lines_(filepath);
  axes = struct();

  i = 1;
  while i <= numel(lines)
      line = lines{i};
      match = regexp(line, '^([XYZxyz])\s+(\d+)\s*:?\s*(.*)$', 'tokens', 'once');
      if isempty(match)
          error('setup_case_from_ldg_lgrid: invalid lgrid header: %s', line);
      end

      axis_name = upper(match{1});
      n_points = str2double(match{2});
      tail = strtrim(match{3});
      if isempty(tail)
          i = i + 1;
          if i > numel(lines)
              error('setup_case_from_ldg_lgrid: missing node list for axis %s.', axis_name);
          end
          tail = lines{i};
      end

      anchors_input = local_parse_braced_list_(tail);
      edges_input = local_expand_axis_points_(anchors_input, n_points);

      axis_struct = struct();
      axis_struct.n_points = n_points;
      axis_struct.anchors_input = anchors_input(:).';
      axis_struct.edges_input = edges_input(:);
      axis_struct.edges = edges_input(:) * length_scale;
      axes.(lower(axis_name)) = axis_struct;

      i = i + 1;
  end

  req = {'x', 'y', 'z'};
  for k = 1:numel(req)
      if ~isfield(axes, req{k})
          error('setup_case_from_ldg_lgrid: lgrid file must define X, Y and Z.');
      end
  end

  grid = struct();
  grid.source = filepath;
  grid.axes = axes;
  grid.x_edges = axes.x.edges;
  grid.y_edges = axes.y.edges;
  grid.z_edges = axes.z.edges;
  grid.Nx = numel(grid.x_edges) - 1;
  grid.Ny = numel(grid.y_edges) - 1;
  grid.Nz = numel(grid.z_edges) - 1;
end

function geom = local_build_geom_(grid)
  geom = struct();
  geom.shape = 'box';
  geom.origin = [grid.x_edges(1), grid.y_edges(1), grid.z_edges(1)];
  geom.L = [grid.x_edges(end) - grid.x_edges(1), ...
            grid.y_edges(end) - grid.y_edges(1), ...
            grid.z_edges(end) - grid.z_edges(1)];
end

function local_validate_layout_vs_grid_(layout, grid)
  if isfield(layout, 'variables_si')
      vars_si = layout.variables_si;
  else
      vars_si = struct();
  end

  if all(isfield(vars_si, {'Lx', 'Ly', 'Lz'}))
      declared = [vars_si.Lx, vars_si.Ly, vars_si.Lz];
      gridded = [grid.x_edges(end) - grid.x_edges(1), ...
                 grid.y_edges(end) - grid.y_edges(1), ...
                 grid.z_edges(end) - grid.z_edges(1)];
      tol = 1e-12 * max(1, max(abs([declared, gridded])));
      if any(abs(declared - gridded) > tol)
          error(['setup_case_from_ldg_lgrid: ldg dimensions [Lx Ly Lz] and lgrid extents ' ...
                 'do not match.']);
      end
  end
end

function layout = local_build_layout_behavior_(layout)
  face_tags = {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'};
  boundary_patches = struct();
  for i = 1:numel(face_tags)
      boundary_patches.(face_tags{i}) = repmat(local_empty_patch_(), 0, 1);
  end

  boundary_rules = layout.rules(arrayfun(@(r) strcmpi(r.location, 'boundary'), layout.rules));
  for i = 1:numel(boundary_rules)
      rule = boundary_rules(i);
      tag = rule.face_tag;
      patch = local_empty_patch_();
      patch.face_tag = tag;
      patch.mode = upper(rule.mode);
      patch.normal = upper(rule.normal);
      patch.bounds = rule.bounds;
      patch.bounds_input = rule.bounds_input;
      patch.patch_area = rule.patch_area;
      patch.raw = rule.raw;
      boundary_patches.(tag)(end + 1, 1) = patch; %#ok<AGROW>
  end

  layout.boundary_patches = boundary_patches;
  layout.warnings = {};
end

function rule = local_finalize_rule_(rule, vars_si)
  tol = 1e-12 * max(1, max(abs(struct2array(vars_si))));
  if isempty(tol) || ~isfinite(tol)
      tol = 1e-12;
  end

  [axis_name, coord] = local_rule_axis_and_coord_(rule);
  rule.axis = axis_name;
  rule.coord = coord;
  rule.patch_area = local_rule_patch_area_(rule);
  [rule.location, rule.face_tag] = local_rule_location_(rule, vars_si, tol);
end

function [axis_name, coord] = local_rule_axis_and_coord_(rule)
  switch upper(rule.normal)
    case {'+X', '-X'}
      axis_name = 'x';
      coord = 0.5 * (rule.bounds(1) + rule.bounds(2));
    case {'+Y', '-Y'}
      axis_name = 'y';
      coord = 0.5 * (rule.bounds(3) + rule.bounds(4));
    case {'+Z', '-Z'}
      axis_name = 'z';
      coord = 0.5 * (rule.bounds(5) + rule.bounds(6));
    otherwise
      error('setup_case_from_ldg_lgrid: invalid plane normal %s', rule.normal);
  end
end

function area = local_rule_patch_area_(rule)
  switch upper(rule.normal)
    case {'+X', '-X'}
      area = max(rule.bounds(4) - rule.bounds(3), 0) * max(rule.bounds(6) - rule.bounds(5), 0);
    case {'+Y', '-Y'}
      area = max(rule.bounds(2) - rule.bounds(1), 0) * max(rule.bounds(6) - rule.bounds(5), 0);
    case {'+Z', '-Z'}
      area = max(rule.bounds(2) - rule.bounds(1), 0) * max(rule.bounds(4) - rule.bounds(3), 0);
    otherwise
      area = 0;
  end
end

function [location, face_tag] = local_rule_location_(rule, vars_si, tol)
  face_tag = '';
  location = 'internal';

  if ~isfield(vars_si, 'Lx') || ~isfield(vars_si, 'Ly') || ~isfield(vars_si, 'Lz')
      return;
  end

  switch upper(rule.normal)
    case '-X'
      if abs(rule.coord - 0) <= tol
          location = 'boundary';
          face_tag = 'x_min';
      end
    case '+X'
      if abs(rule.coord - vars_si.Lx) <= tol
          location = 'boundary';
          face_tag = 'x_max';
      end
    case '-Y'
      if abs(rule.coord - 0) <= tol
          location = 'boundary';
          face_tag = 'y_min';
      end
    case '+Y'
      if abs(rule.coord - vars_si.Ly) <= tol
          location = 'boundary';
          face_tag = 'y_max';
      end
    case '-Z'
      if abs(rule.coord - 0) <= tol
          location = 'boundary';
          face_tag = 'z_min';
      end
    case '+Z'
      if abs(rule.coord - vars_si.Lz) <= tol
          location = 'boundary';
          face_tag = 'z_max';
      end
  end
end

function values = local_parse_braced_list_(line)
  match = regexp(line, '^\{(.*)\}$', 'tokens', 'once');
  if isempty(match)
      error('setup_case_from_ldg_lgrid: expected braced list, got: %s', line);
  end

  parts = regexp(match{1}, '\s*,\s*', 'split');
  values = zeros(1, numel(parts));
  for i = 1:numel(parts)
      values(i) = local_eval_expr_(parts{i}, struct());
  end
end

function points = local_expand_axis_points_(anchors, n_points)
  anchors = anchors(:).';
  if n_points < 2
      error('setup_case_from_ldg_lgrid: axis needs at least two grid points.');
  end
  if numel(anchors) < 2
      error('setup_case_from_ldg_lgrid: axis needs at least two anchors.');
  end
  if any(diff(anchors) <= 0)
      error('setup_case_from_ldg_lgrid: axis anchors must be strictly increasing.');
  end

  if n_points == numel(anchors)
      points = anchors;
      return;
  end

  uniform_points = linspace(anchors(1), anchors(end), n_points);
  tol = 1e-12 * max(1, abs(anchors(end) - anchors(1)));
  if all(arrayfun(@(a) any(abs(uniform_points - a) <= tol), anchors))
      points = uniform_points;
      return;
  end

  n_seg = numel(anchors) - 1;
  n_intervals = n_points - 1;
  if n_intervals < n_seg
      error('setup_case_from_ldg_lgrid: grid point count is too small to include all anchors.');
  end

  lengths = diff(anchors);
  exact = lengths / sum(lengths) * n_intervals;
  counts = floor(exact);
  remainder = n_intervals - sum(counts);
  if remainder > 0
      [~, order] = sort(exact - counts, 'descend');
      counts(order(1:remainder)) = counts(order(1:remainder)) + 1;
  end

  while any(counts == 0)
      zero_idx = find(counts == 0, 1, 'first');
      donors = find(counts > 1);
      if isempty(donors)
          error('setup_case_from_ldg_lgrid: failed to allocate intervals for all anchors.');
      end
      [~, donor_ord] = max(counts(donors) - exact(donors));
      donor = donors(donor_ord);
      counts(donor) = counts(donor) - 1;
      counts(zero_idx) = 1;
  end

  points = anchors(1);
  for i = 1:n_seg
      seg_points = linspace(anchors(i), anchors(i + 1), counts(i) + 1);
      points = [points, seg_points(2:end)]; %#ok<AGROW>
  end
end

function value = local_eval_expr_(expr, vars)
  expr = strtrim(expr);
  var_names = regexp(expr, '\$([A-Za-z]\w*)\$', 'tokens');
  for i = 1:numel(var_names)
      name = var_names{i}{1};
      if ~isfield(vars, name)
          error('setup_case_from_ldg_lgrid: undefined variable $%s$ in expression "%s".', name, expr);
      end
      expr = regexprep(expr, ['\$' name '\$'], sprintf('%.17g', vars.(name)));
  end

  if isempty(regexp(expr, '^[0-9eE\+\-\*\/\.\(\)\s]+$', 'once'))
      error('setup_case_from_ldg_lgrid: unsafe expression "%s".', expr);
  end

  value = eval(expr); %#ok<EVLDIR>
  if ~(isscalar(value) && isfinite(value))
      error('setup_case_from_ldg_lgrid: expression did not evaluate to a finite scalar: %s', expr);
  end
end

function lines = local_read_clean_lines_(filepath)
  raw = string(splitlines(fileread(filepath)));
  lines = {};
  for i = 1:numel(raw)
      line = char(raw(i));
      line = strrep(line, char(65279), '');
      line = regexprep(line, '[#%].*$', '');
      line = strtrim(line);
      if ~isempty(line)
          lines{end + 1, 1} = line; %#ok<AGROW>
      end
  end
end

function local_print_case_summary_(cs, length_scale, geom)
  Lin = cs.geom.L / length_scale;
  fprintf('[case] loaded %s and %s\n', cs.layout.source, cs.grid.source);
  fprintf('[geom] box | L = (%.6g, %.6g, %.6g) %s\n', ...
      Lin(1), Lin(2), Lin(3), cs.units.input_length);
  fprintf('[mesh] cells = (%d, %d, %d) | nodes = (%d, %d, %d)\n', ...
      cs.grid.Nx, cs.grid.Ny, cs.grid.Nz, ...
      cs.grid.axes.x.n_points, cs.grid.axes.y.n_points, cs.grid.axes.z.n_points);
  if ~isempty(cs.materials)
      fprintf('[mat] regions = %d | materials = %s\n', numel(cs.regions), strjoin(cs.materials, ', '));
  end
  if isfield(cs.layout, 'reservoirs') && ~isempty(cs.layout.reservoirs)
      fprintf('[reservoir] count = %d\n', numel(cs.layout.reservoirs));
  end

  faces = {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'};
  for i = 1:numel(faces)
      tag = faces{i};
      patches = cs.layout.boundary_patches.(tag);
      coverage = local_boundary_patch_coverage_(patches, geom, tag);
      fprintf('[face] %-5s patches=%d coverage=%.1f%%', tag, numel(patches), 100 * coverage);
      fprintf('\n');
  end

  for i = 1:numel(cs.layout.warnings)
      fprintf('[layout] %s\n', cs.layout.warnings{i});
  end
end

function reg = local_empty_region_()
  reg = struct('bounds_input', zeros(1, 6), ...
               'bounds', zeros(1, 6), ...
               'material', '', ...
               'raw', '');
end

function rule = local_empty_rule_()
  rule = struct('kind', '', ...
                'bounds_input', zeros(1, 6), ...
                'bounds', zeros(1, 6), ...
                'normal', '', ...
                'mode', '', ...
                'axis', '', ...
                'coord', NaN, ...
                'patch_area', 0, ...
                'location', 'internal', ...
                'face_tag', '', ...
                'raw', '');
end

function patch = local_empty_patch_()
  patch = struct('face_tag', '', ...
                 'mode', '', ...
                 'normal', '', ...
                 'bounds', zeros(1, 6), ...
                 'bounds_input', zeros(1, 6), ...
                 'patch_area', 0, ...
                 'raw', '');
end

function res = local_empty_reservoir_()
  res = struct('id', int32(0), ...
               'name', '', ...
               'bounds_input', zeros(1, 6), ...
               'bounds', zeros(1, 6), ...
               'raw', '');
end

function coverage = local_boundary_patch_coverage_(patches, geom, face_tag)
  if isempty(patches)
      coverage = 0;
      return;
  end
  total_area = 0;
  for i = 1:numel(patches)
      total_area = total_area + patches(i).patch_area;
  end
  coverage = total_area / max(local_face_area_(geom, face_tag), eps);
end

function area = local_face_area_(geom, face_tag)
  switch lower(face_tag)
    case {'x_min', 'x_max'}
      area = geom.L(2) * geom.L(3);
    case {'y_min', 'y_max'}
      area = geom.L(1) * geom.L(3);
    case {'z_min', 'z_max'}
      area = geom.L(1) * geom.L(2);
    otherwise
      error('setup_case_from_ldg_lgrid: invalid face tag %s', face_tag);
  end
end
