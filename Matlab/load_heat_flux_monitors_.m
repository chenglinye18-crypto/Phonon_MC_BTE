function [monitors, warnings] = load_heat_flux_monitors_(mesh, output_cfg)
% load_heat_flux_monitors_ Load custom heat-flux monitor planes from text.
%
% Text format:
%   x0 x1 y0 y1 z0 z1 direction [label]
%
% Lengths are interpreted using output_cfg.monitor_length_scale.

  monitors = repmat(local_empty_monitor_(), 0, 1);
  warnings = {};

  if ~isstruct(output_cfg) || ~isfield(output_cfg, 'heat_flux_monitor_file') || ...
          isempty(output_cfg.heat_flux_monitor_file)
      return;
  end

  filepath = char(output_cfg.heat_flux_monitor_file);
  if exist(filepath, 'file') ~= 2
      warnings{end + 1} = sprintf('heat flux monitor file not found: %s', filepath); %#ok<AGROW>
      return;
  end

  length_scale = 1;
  if isfield(output_cfg, 'monitor_length_scale') && isfinite(output_cfg.monitor_length_scale)
      length_scale = output_cfg.monitor_length_scale;
  end

  lines = local_read_clean_lines_(filepath);
  for i = 1:numel(lines)
      tokens = regexp(lines{i}, '[,\s]+', 'split');
      tokens = tokens(~cellfun(@isempty, tokens));
      if numel(tokens) < 7
          error('load_heat_flux_monitors_: invalid monitor line: %s', lines{i});
      end

      bounds_input = str2double(tokens(1:6));
      if any(~isfinite(bounds_input))
          error('load_heat_flux_monitors_: invalid numeric bounds in line: %s', lines{i});
      end

      requested_direction = upper(tokens{7});
      if numel(tokens) >= 8
          label = tokens{8};
      else
          label = sprintf('monitor_%03d', i);
      end

      bounds = bounds_input * length_scale;
      [plane_axis, coord, area] = local_monitor_plane_(bounds);
      sign_char = '+';
      if ~isempty(requested_direction) && requested_direction(1) == '-'
          sign_char = '-';
      end
      effective_normal = [sign_char upper(plane_axis)];

      monitor = local_empty_monitor_();
      monitor.id = i;
      monitor.label = label;
      monitor.bounds_input = bounds_input;
      monitor.bounds = bounds;
      monitor.requested_direction = requested_direction;
      monitor.axis = plane_axis;
      monitor.coord = coord;
      monitor.area = area;
      monitor.effective_normal = effective_normal;
      monitor.raw = lines{i};

      mismatch_msg = local_direction_mismatch_(requested_direction, plane_axis, label);
      if ~isempty(mismatch_msg)
          warnings{end + 1} = mismatch_msg; %#ok<AGROW>
          monitor.warning = mismatch_msg;
      end

      if isfield(mesh, 'domain_box') && numel(mesh.domain_box) == 6
          if ~local_bounds_inside_domain_(bounds, mesh.domain_box)
              warn = sprintf('monitor "%s" extends outside domain bounds.', label);
              warnings{end + 1} = warn; %#ok<AGROW>
              monitor.warning = strtrim(strjoin({monitor.warning, warn}, ' | '));
          end
      end

      monitors(end + 1, 1) = monitor; %#ok<AGROW>
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

function [axis_name, coord, area] = local_monitor_plane_(bounds)
  tol = 1e-15 * max(1, max(abs(bounds)));
  fixed = [abs(bounds(2) - bounds(1)) <= tol, ...
           abs(bounds(4) - bounds(3)) <= tol, ...
           abs(bounds(6) - bounds(5)) <= tol];
  if nnz(fixed) ~= 1
      error('load_heat_flux_monitors_: each monitor must define exactly one plane coordinate.');
  end

  if fixed(1)
      axis_name = 'x';
      coord = 0.5 * (bounds(1) + bounds(2));
      area = max(bounds(4) - bounds(3), 0) * max(bounds(6) - bounds(5), 0);
  elseif fixed(2)
      axis_name = 'y';
      coord = 0.5 * (bounds(3) + bounds(4));
      area = max(bounds(2) - bounds(1), 0) * max(bounds(6) - bounds(5), 0);
  else
      axis_name = 'z';
      coord = 0.5 * (bounds(5) + bounds(6));
      area = max(bounds(2) - bounds(1), 0) * max(bounds(4) - bounds(3), 0);
  end
end

function tf = local_bounds_inside_domain_(bounds, domain_box)
  tf = bounds(1) >= domain_box(1) && bounds(2) <= domain_box(2) && ...
       bounds(3) >= domain_box(3) && bounds(4) <= domain_box(4) && ...
       bounds(5) >= domain_box(5) && bounds(6) <= domain_box(6);
end

function msg = local_direction_mismatch_(requested_direction, plane_axis, label)
  msg = '';
  if numel(requested_direction) < 2
      return;
  end
  requested_axis = upper(requested_direction(end));
  if requested_axis ~= upper(plane_axis)
      msg = sprintf('monitor "%s": requested direction %s does not match plane normal axis %s; using %s.', ...
          label, requested_direction, upper(plane_axis), [requested_direction(1) upper(plane_axis)]);
  end
end

function monitor = local_empty_monitor_()
  monitor = struct('id', 0, ...
                   'label', '', ...
                   'bounds_input', zeros(1, 6), ...
                   'bounds', zeros(1, 6), ...
                   'requested_direction', '', ...
                   'axis', '', ...
                   'coord', NaN, ...
                   'area', NaN, ...
                   'effective_normal', '', ...
                   'warning', '', ...
                   'raw', '');
end
