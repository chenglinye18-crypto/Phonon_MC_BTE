function [state, info] = refresh_reservoir_particles_(state, mesh, spec, opts)
% refresh_reservoir_particles_ Reset reservoir-cell particles to equilibrium.

  info = struct('refreshed', false, ...
                'cell_ids', zeros(0, 1), ...
                'removed_particles', 0, ...
                'added_particles', 0, ...
                'target_temperature_cell', zeros(0, 1), ...
                'reference_temperature_cell', zeros(0, 1));

  if ~isfield(mesh, 'reservoirs') || isempty(mesh.reservoirs)
      return;
  end

  all_cells = local_collect_reservoir_cells_(mesh.reservoirs);
  if isempty(all_cells)
      return;
  end

  target_T_all = local_target_temperature_field_(state, mesh, opts);
  Tref_all = local_reference_temperature_field_(state, mesh, opts, target_T_all);
  target_T = target_T_all(all_cells);
  Tref = Tref_all(all_cells);

  kept = state.p;
  removed_particles = 0;
  if ~isempty(state.p)
      p_cells = double([state.p.cell].');
      keep_mask = ~ismember(p_cells, all_cells);
      removed_particles = nnz(~keep_mask);
      kept = state.p(keep_mask);
  end

  next_id = local_next_particle_id_(kept);
  [newp, sample_info] = sample_equilibrium_particles_for_cells_( ...
      mesh, spec, opts, all_cells, target_T, Tref, next_id);

  if isempty(kept)
      state.p = newp;
  elseif isempty(newp)
      state.p = kept;
  else
      state.p = [kept; newp];
  end

  Nc = local_infer_Nc_(mesh);
  if isempty(state.p)
      state.Nsp_cell = zeros(Nc, 1);
  else
      state.Nsp_cell = accumarray(double([state.p.cell].'), 1, [Nc 1], @sum, 0);
  end

  if ~isfield(state, 'info') || ~isstruct(state.info)
      state.info = struct();
  end
  mask = false(Nc, 1);
  mask(all_cells) = true;
  state.info.reservoir_cell_mask = mask;
  state.info.reservoir_target_temperature_cell = target_T_all;
  state.info.reservoir_reference_temperature_cell = Tref_all;
  state.info.reservoir_last_refresh_particles = sample_info.Nsp_tot;

  info.refreshed = true;
  info.cell_ids = all_cells;
  info.removed_particles = removed_particles;
  info.added_particles = sample_info.Nsp_tot;
  info.target_temperature_cell = target_T;
  info.reference_temperature_cell = Tref;
end

function cells = local_collect_reservoir_cells_(reservoirs)
  cells = zeros(0, 1);
  for i = 1:numel(reservoirs)
      cells = [cells; reservoirs(i).cell_ids(:)]; %#ok<AGROW>
  end
  cells = unique(cells(:), 'stable');
end

function Tcell = local_target_temperature_field_(state, mesh, opts)
  Nc = local_infer_Nc_(mesh);

  if isstruct(state) && isfield(state, 'info') && isstruct(state.info)
      if isfield(state.info, 'reservoir_target_temperature_cell') && ...
              numel(state.info.reservoir_target_temperature_cell) == Nc
          Tcell = state.info.reservoir_target_temperature_cell(:);
          return;
      end
      if isfield(state.info, 'T_init_cell') && numel(state.info.T_init_cell) == Nc
          Tcell = state.info.T_init_cell(:);
          return;
      end
  end

  fallback_T = NaN;
  if isfield(opts, 'T0') && isfinite(opts.T0)
      fallback_T = opts.T0;
  elseif isfield(opts, 'Tref') && isfinite(opts.Tref)
      fallback_T = opts.Tref;
  end
  [Tcell, ~] = load_initial_temperature_field_(mesh, opts, fallback_T);
end

function Tref_cell = local_reference_temperature_field_(state, mesh, opts, target_T)
  Nc = local_infer_Nc_(mesh);

  if isstruct(state) && isfield(state, 'info') && isstruct(state.info)
      if isfield(state.info, 'reservoir_reference_temperature_cell') && ...
              numel(state.info.reservoir_reference_temperature_cell) == Nc
          Tref_cell = state.info.reservoir_reference_temperature_cell(:);
          return;
      end
      if isfield(state.info, 'Tref_cell') && numel(state.info.Tref_cell) == Nc
          Tref_cell = state.info.Tref_cell(:);
          return;
      end
  end

  default_Tref = NaN;
  if isfield(opts, 'Tref') && isfinite(opts.Tref)
      default_Tref = opts.Tref;
  elseif ~isempty(target_T)
      default_Tref = mean(target_T);
  end
  [Tref_cell, ~] = load_reference_temperature_field_(mesh, opts, default_Tref);
end

function nid = local_next_particle_id_(p)
  if isempty(p)
      nid = 0;
  else
      nid = max([p.id]);
  end
end

function Nc = local_infer_Nc_(mesh)
  if isfield(mesh, 'Nc') && ~isempty(mesh.Nc)
      Nc = mesh.Nc;
  else
      Nc = mesh.Nx * mesh.Ny * mesh.Nz;
  end
end
