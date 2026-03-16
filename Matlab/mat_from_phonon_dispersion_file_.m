function mat = mat_from_phonon_dispersion_file_(varargin)
% mat_from_phonon_dispersion_file_ Build a table-driven material from text data.
%
% Required file columns:
%   branch_id, q_real(1/m), f(THz), vg(m/s)
%
% Optional comment metadata:
%   # branch_names=TA,TA,LA; degeneracy=1,1,1
%
% The number of branches is inferred from unique branch ids in the file.

  parser = inputParser;
  parser.FunctionName = mfilename;
  addParameter(parser, 'FilePath', '', @(x) ischar(x) || isstring(x));
  addParameter(parser, 'MaterialName', 'TableDriven', @(x) ischar(x) || isstring(x));
  addParameter(parser, 'BranchNames', {}, @(x) iscell(x) || isstring(x));
  addParameter(parser, 'Degeneracy', [], @isnumeric);
  parse(parser, varargin{:});

  filepath = char(parser.Results.FilePath);
  if isempty(filepath)
      error('mat_from_phonon_dispersion_file_: FilePath is required.');
  end
  if exist(filepath, 'file') ~= 2
      error('mat_from_phonon_dispersion_file_: file not found: %s', filepath);
  end

  header_meta = local_parse_header_metadata_(filepath);
  data = readmatrix(filepath, 'FileType', 'text', 'CommentStyle', '#');
  if isempty(data) || size(data, 2) < 4
      error('mat_from_phonon_dispersion_file_: %s must contain at least four numeric columns.', filepath);
  end
  data = data(:, 1:4);
  data = data(all(isfinite(data), 2), :);
  if isempty(data)
      error('mat_from_phonon_dispersion_file_: %s contains no valid numeric rows.', filepath);
  end

  branch_id = round(data(:, 1));
  q_raw = data(:, 2);
  f_thz = data(:, 3);
  vg_raw = data(:, 4);

  branch_ids = unique(branch_id, 'stable');
  B = numel(branch_ids);
  if B < 1
      error('mat_from_phonon_dispersion_file_: no branches found in %s.', filepath);
  end

  q_common = unique(q_raw(:), 'sorted');
  q_common = q_common(:).';
  M = numel(q_common);
  omega_tab = zeros(B, M);
  vg_tab = zeros(B, M);

  n_negative_freq = 0;
  for ib = 1:B
      mask = branch_id == branch_ids(ib);
      q_b = q_raw(mask);
      f_b = f_thz(mask);
      vg_b = vg_raw(mask);

      [q_b, order] = sort(q_b(:));
      f_b = f_b(order);
      vg_b = vg_b(order);
      [q_b, f_b, vg_b] = local_unique_samples_(q_b, f_b, vg_b);

      n_negative_freq = n_negative_freq + nnz(f_b < 0);
      omega_b = 2 * pi * 1e12 * max(f_b, 0);

      omega_tab(ib, :) = local_interp_clamped_(q_common, q_b, omega_b);
      vg_tab(ib, :) = local_interp_clamped_(q_common, q_b, vg_b);
  end

  branch_names = parser.Results.BranchNames;
  if isempty(branch_names)
      if ~isempty(header_meta.branch_names)
          branch_names = header_meta.branch_names;
      else
          branch_names = arrayfun(@(id) sprintf('B%d', id), branch_ids, 'UniformOutput', false);
      end
  else
      if isstring(branch_names), branch_names = cellstr(branch_names); end
  end
  if numel(branch_names) ~= B
      error('mat_from_phonon_dispersion_file_: BranchNames count (%d) does not match branch count (%d).', ...
          numel(branch_names), B);
  end
  branch_names = reshape(cellstr(string(branch_names)), 1, []);

  degeneracy = parser.Results.Degeneracy;
  if isempty(degeneracy)
      if ~isempty(header_meta.degeneracy)
          degeneracy = header_meta.degeneracy;
      else
          degeneracy = ones(1, B);
      end
  else
      degeneracy = reshape(degeneracy, 1, []);
  end
  if numel(degeneracy) ~= B
      error('mat_from_phonon_dispersion_file_: Degeneracy count (%d) does not match branch count (%d).', ...
          numel(degeneracy), B);
  end

  material_name = char(parser.Results.MaterialName);
  mat = struct();
  mat.name = material_name;
  mat.source_file = filepath;
  mat.branch_ids = branch_ids(:).';
  mat.branch_names = branch_names;
  mat.degeneracy = degeneracy;
  mat.B = B;
  mat.q = q_common;
  mat.qmax = max(q_common);
  mat.omega_tab = omega_tab;
  mat.vg_tab = vg_tab;
  mat.frequency_THz_tab = omega_tab / (2 * pi * 1e12);
  mat.n_negative_freq_entries = n_negative_freq;

  mat.omega = @(b, qq) local_interp_clamped_(qq, q_common, omega_tab(b, :));
  mat.vg = @(b, qq) local_interp_clamped_(qq, q_common, vg_tab(b, :));
  mat.omega_all = @(qq) cell2mat(arrayfun(@(b) mat.omega(b, qq), 1:B, 'UniformOutput', false).');
  mat.vg_all = @(qq) cell2mat(arrayfun(@(b) mat.vg(b, qq), 1:B, 'UniformOutput', false).');

  hbar_meVs = 6.582119569e-13;
  mat.energy_meV = @(b, qq) hbar_meVs .* mat.omega(b, qq);

  if n_negative_freq > 0
      warning('mat_from_phonon_dispersion_file_: clamped %d negative frequency samples to zero in %s.', ...
          n_negative_freq, filepath);
  end
end

function meta = local_parse_header_metadata_(filepath)
  meta = struct('branch_names', {{}}, 'degeneracy', []);

  lines = string(splitlines(fileread(filepath)));
  for i = 1:numel(lines)
      line = strtrim(char(lines(i)));
      if isempty(line)
          continue;
      end
      if line(1) ~= '#'
          break;
      end

      text = strtrim(line(2:end));
      if isempty(text)
          continue;
      end

      branch_names = local_extract_list_token_(text, 'branch_names');
      if ~isempty(branch_names)
          meta.branch_names = branch_names;
      end

      degeneracy = local_extract_numeric_list_token_(text, 'degeneracy');
      if ~isempty(degeneracy)
          meta.degeneracy = degeneracy;
      end
  end
end

function values = local_extract_list_token_(text, key)
  values = {};
  token = regexp(text, [key '\s*=\s*([^;#]+)'], 'tokens', 'once');
  if isempty(token)
      return;
  end

  raw = strtrim(token{1});
  if isempty(raw)
      return;
  end

  parts = regexp(raw, '\s*,\s*', 'split');
  parts = parts(~cellfun(@isempty, parts));
  values = cellfun(@strtrim, parts, 'UniformOutput', false);
end

function values = local_extract_numeric_list_token_(text, key)
  values = [];
  parts = local_extract_list_token_(text, key);
  if isempty(parts)
      return;
  end

  values = str2double(parts);
  if any(~isfinite(values))
      error('mat_from_phonon_dispersion_file_: invalid numeric metadata "%s".', key);
  end
  values = reshape(values, 1, []);
end

function [q_u, f_u, vg_u] = local_unique_samples_(q, f, vg)
  [q_u, ~, ic] = unique(q, 'stable');
  if numel(q_u) == numel(q)
      f_u = f;
      vg_u = vg;
      return;
  end

  f_u = accumarray(ic, f, [], @mean);
  vg_u = accumarray(ic, vg, [], @mean);
end

function yq = local_interp_clamped_(xq, x, y)
  x = x(:);
  y = y(:);
  xq_shape = size(xq);
  xq = xq(:);

  if numel(x) == 1
      yq = repmat(y, size(xq));
      yq = reshape(yq, xq_shape);
      return;
  end

  xq_clamped = min(max(xq, x(1)), x(end));
  yq = interp1(x, y, xq_clamped, 'pchip');
  yq = reshape(yq, xq_shape);
end
