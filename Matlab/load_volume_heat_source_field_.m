function [qvol, meta] = load_volume_heat_source_field_(mesh, opts, default_qvol)
% load_volume_heat_source_field_ Load per-cell volumetric heat source from text.
%
% Text format:
%   idxcell,idycell,idzcell,HeatSource

  if nargin < 3 || isempty(default_qvol)
      default_qvol = 0;
  end

  Nc = mesh.Nx * mesh.Ny * mesh.Nz;
  if isscalar(default_qvol)
      qvol = default_qvol * ones(Nc, 1);
  else
      qvol = default_qvol(:);
      if numel(qvol) ~= Nc
          error('load_volume_heat_source_field_: default_qvol must be scalar or Nc-by-1.');
      end
  end

  meta = struct('source', '', 'used_file', false, 'q_min', min(qvol), 'q_mean', mean(qvol), 'q_max', max(qvol));

  if ~isfield(opts, 'volume_heat_source_file') || isempty(opts.volume_heat_source_file)
      return;
  end

  filepath = opts.volume_heat_source_file;
  data = readmatrix(filepath, 'FileType', 'text');
  if size(data, 2) < 4
      error('load_volume_heat_source_field_: file %s must have four columns.', filepath);
  end

  idx = data(:, 1);
  idy = data(:, 2);
  idz = data(:, 3);
  qsrc = data(:, 4);

  valid = isfinite(idx) & isfinite(idy) & isfinite(idz) & isfinite(qsrc);
  idx = idx(valid);
  idy = idy(valid);
  idz = idz(valid);
  qsrc = qsrc(valid);

  if isempty(idx)
      error('load_volume_heat_source_field_: file %s contains no valid numeric rows.', filepath);
  end

  idx = round(idx);
  idy = round(idy);
  idz = round(idz);
  if any(idx < 1 | idx > mesh.Nx | idy < 1 | idy > mesh.Ny | idz < 1 | idz > mesh.Nz)
      error('load_volume_heat_source_field_: indices in %s exceed mesh bounds.', filepath);
  end

  lin = sub2ind([mesh.Nx, mesh.Ny, mesh.Nz], idx, idy, idz);
  if numel(unique(lin)) ~= numel(lin)
      error('load_volume_heat_source_field_: duplicate cell indices found in %s.', filepath);
  end

  qvol(lin) = qsrc;
  if numel(lin) ~= Nc
      error('load_volume_heat_source_field_: %s does not cover all mesh cells.', filepath);
  end

  meta.source = filepath;
  meta.used_file = true;
  meta.q_min = min(qvol);
  meta.q_mean = mean(qvol);
  meta.q_max = max(qvol);
end
