function [Tref_cell, meta] = load_reference_temperature_field_(mesh, opts, default_Tref)
% load_reference_temperature_field_ Load per-cell reference temperature from text.
%
% Text format:
%   idxcell,idycell,idzcell,Tref

  if nargin < 3 || isempty(default_Tref)
      default_Tref = NaN;
  end

  Nc = mesh.Nx * mesh.Ny * mesh.Nz;
  if isfinite(default_Tref)
      Tref_cell = default_Tref * ones(Nc, 1);
  else
      Tref_cell = NaN(Nc, 1);
  end

  meta = struct('source', '', 'used_file', false, 'T_min', NaN, 'T_mean', NaN, 'T_max', NaN);

  if ~isfield(opts, 'reference_temperature_file') || isempty(opts.reference_temperature_file)
      if any(~isfinite(Tref_cell))
          error('load_reference_temperature_field_: missing reference_temperature_file and no default_Tref provided.');
      end
      meta.T_min = min(Tref_cell);
      meta.T_mean = mean(Tref_cell);
      meta.T_max = max(Tref_cell);
      return;
  end

  filepath = opts.reference_temperature_file;
  data = readmatrix(filepath, 'FileType', 'text');
  if size(data, 2) < 4
      error('load_reference_temperature_field_: file %s must have four columns.', filepath);
  end

  idx = data(:, 1);
  idy = data(:, 2);
  idz = data(:, 3);
  tref = data(:, 4);

  valid = isfinite(idx) & isfinite(idy) & isfinite(idz) & isfinite(tref);
  idx = idx(valid);
  idy = idy(valid);
  idz = idz(valid);
  tref = tref(valid);
  if isempty(idx)
      error('load_reference_temperature_field_: file %s contains no valid numeric rows.', filepath);
  end

  idx = round(idx);
  idy = round(idy);
  idz = round(idz);
  if any(idx < 1 | idx > mesh.Nx | idy < 1 | idy > mesh.Ny | idz < 1 | idz > mesh.Nz)
      error('load_reference_temperature_field_: indices in %s exceed mesh bounds.', filepath);
  end

  lin = sub2ind([mesh.Nx, mesh.Ny, mesh.Nz], idx, idy, idz);
  if numel(unique(lin)) ~= numel(lin)
      error('load_reference_temperature_field_: duplicate cell indices found in %s.', filepath);
  end

  Tref_cell(lin) = tref;
  if any(~isfinite(Tref_cell))
      error('load_reference_temperature_field_: %s does not cover all mesh cells.', filepath);
  end

  meta.source = filepath;
  meta.used_file = true;
  meta.T_min = min(Tref_cell);
  meta.T_mean = mean(Tref_cell);
  meta.T_max = max(Tref_cell);
end
