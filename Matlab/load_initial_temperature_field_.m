function [Tcell, meta] = load_initial_temperature_field_(mesh, opts, default_T)
% load_initial_temperature_field_ Load per-cell initial temperature from CSV.
%
% CSV format:
%   idxcell,idycell,idzcell,Temperature

  if nargin < 3 || isempty(default_T)
      default_T = NaN;
  end

  Nc = mesh.Nx * mesh.Ny * mesh.Nz;
  if isfinite(default_T)
      Tcell = default_T * ones(Nc, 1);
  else
      Tcell = NaN(Nc, 1);
  end

  meta = struct('source', '', 'used_file', false, 'T_min', NaN, 'T_mean', NaN, 'T_max', NaN);

  if ~isfield(opts, 'initial_temperature_file') || isempty(opts.initial_temperature_file)
      if any(~isfinite(Tcell))
          error('load_initial_temperature_field_: missing initial_temperature_file and no default_T provided.');
      end
      meta.T_min = min(Tcell);
      meta.T_mean = mean(Tcell);
      meta.T_max = max(Tcell);
      return;
  end

  filepath = opts.initial_temperature_file;
  data = readmatrix(filepath);
  if size(data, 2) < 4
      error('load_initial_temperature_field_: file %s must have four columns.', filepath);
  end

  idx = data(:, 1);
  idy = data(:, 2);
  idz = data(:, 3);
  temp = data(:, 4);

  if any(~isfinite(idx) | ~isfinite(idy) | ~isfinite(idz) | ~isfinite(temp))
      error('load_initial_temperature_field_: file %s contains non-finite values.', filepath);
  end

  idx = round(idx);
  idy = round(idy);
  idz = round(idz);
  if any(idx < 1 | idx > mesh.Nx | idy < 1 | idy > mesh.Ny | idz < 1 | idz > mesh.Nz)
      error('load_initial_temperature_field_: indices in %s exceed mesh bounds.', filepath);
  end

  lin = sub2ind([mesh.Nx, mesh.Ny, mesh.Nz], idx, idy, idz);
  if numel(unique(lin)) ~= numel(lin)
      error('load_initial_temperature_field_: duplicate cell indices found in %s.', filepath);
  end

  Tcell(lin) = temp;
  if any(~isfinite(Tcell))
      error('load_initial_temperature_field_: %s does not cover all mesh cells.', filepath);
  end

  meta.source = filepath;
  meta.used_file = true;
  meta.T_min = min(Tcell);
  meta.T_mean = mean(Tcell);
  meta.T_max = max(Tcell);
end
