function write_periodic_output_(out_cfg, mesh, spec, state, Tprime, step, elapsed_time, dt_step)
% write_periodic_output_ Write periodic solver statistics for one step.

  if ~isstruct(out_cfg) || ~isfield(out_cfg, 'enabled') || ~out_cfg.enabled
      return;
  end

  if nargin < 8 || isempty(dt_step)
      dt_step = NaN;
  end

  step_dir = local_prepare_step_dir_(out_cfg, step);
  wall_clock_elapsed = NaN;
  if isfield(out_cfg, 'run_wallclock_tic') && ~isempty(out_cfg.run_wallclock_tic)
      wall_clock_elapsed = toc(out_cfg.run_wallclock_tic);
  end

  local_write_temperature_(step_dir, mesh, Tprime);
  local_write_branch_stats_(step_dir, spec, state);
  local_write_heat_flux_(step_dir, out_cfg, elapsed_time);
  local_write_step_info_(step_dir, state, Tprime, step, dt_step, elapsed_time, wall_clock_elapsed, out_cfg.interval_time);
  local_append_step_history_(out_cfg, state, Tprime, step, dt_step, elapsed_time, wall_clock_elapsed);
end

function step_dir = local_prepare_step_dir_(out_cfg, step)
  step_dir = fullfile(out_cfg.steps_dir, sprintf('step_%05d', step));
  if exist(step_dir, 'dir') ~= 7
      mkdir(step_dir);
  end
end

function local_write_temperature_(step_dir, mesh, Tprime)
  Nx = mesh.Nx;
  Ny = mesh.Ny;
  Nz = mesh.Nz;
  [I, J, K] = ndgrid(1:Nx, 1:Ny, 1:Nz);
  data = [I(:), J(:), K(:), Tprime(:)];
  rows = [{'idxcell', 'idycell', 'idzcell', 'Temperature'}; num2cell(data)];
  filepath = fullfile(step_dir, 'temperature.txt');
  writecell(rows, filepath, 'Delimiter', ',');
end

function local_write_branch_stats_(step_dir, spec, state)
  B = spec.B;
  rows = {'branch_id', 'branch_name', 'superparticle_count', 'phonon_count_net', ...
          'phonon_count_abs', 'energy_net_J', 'energy_abs_J'};

  if isempty(state.p)
      for b = 1:B
          rows(end + 1, :) = {b, spec.branches{b}, 0, 0, 0, 0, 0}; %#ok<AGROW>
      end
  else
      hbar = 1.054571817e-34;
      b_all = double([state.p.b].');
      w_all = [state.p.w].';
      E_all = local_particle_energy_vec_(state);
      n_net = E_all ./ (hbar * max(w_all, 1e-30));
      n_abs = abs(E_all) ./ (hbar * max(w_all, 1e-30));

      for b = 1:B
          mask = b_all == b;
          rows(end + 1, :) = {b, spec.branches{b}, nnz(mask), ... %#ok<AGROW>
                              sum(n_net(mask)), sum(n_abs(mask)), ...
                              sum(E_all(mask)), sum(abs(E_all(mask)))};
      end
  end

  filepath = fullfile(step_dir, 'branch_stats.txt');
  writecell(rows, filepath, 'Delimiter', ',');
end

function local_write_heat_flux_(step_dir, out_cfg, elapsed_time)
  rows = {'label', 'requested_direction', 'effective_normal', 'area_m2', 'elapsed_time_s', ...
          'interval_energy_net_J', 'cumulative_energy_net_J', ...
          'flux_interval_W_m2', 'flux_cumulative_W_m2', ...
          'interval_crossings_pos', 'interval_crossings_neg', ...
          'cumulative_crossings_pos', 'cumulative_crossings_neg', 'warning'};

  if isempty(out_cfg.monitors)
      filepath = fullfile(step_dir, 'heat_flux.txt');
      writecell(rows, filepath, 'Delimiter', ',');
      return;
  end

  for i = 1:numel(out_cfg.monitors)
      m = out_cfg.monitors(i);
      flux_interval = NaN;
      if out_cfg.interval_time > 0 && m.area > 0
          flux_interval = out_cfg.interval_energy(i) / (m.area * out_cfg.interval_time);
      end

      flux_cumulative = NaN;
      if elapsed_time > 0 && m.area > 0
          flux_cumulative = out_cfg.cum_energy(i) / (m.area * elapsed_time);
      end

      rows(end + 1, :) = {m.label, m.requested_direction, m.effective_normal, m.area, ... %#ok<AGROW>
                          elapsed_time, out_cfg.interval_energy(i), out_cfg.cum_energy(i), ...
                          flux_interval, flux_cumulative, ...
                          out_cfg.interval_crossings_pos(i), out_cfg.interval_crossings_neg(i), ...
                          out_cfg.cum_crossings_pos(i), out_cfg.cum_crossings_neg(i), m.warning};
  end

  filepath = fullfile(step_dir, 'heat_flux.txt');
  writecell(rows, filepath, 'Delimiter', ',');
end

function local_write_step_info_(step_dir, state, Tprime, step, dt_step, elapsed_time, wall_clock_elapsed, interval_time)
  rows = {
      'step', 'dt_s', 'elapsed_time_s', 'interval_time_s', 'wall_clock_elapsed_s', ...
      'Np', 'T_min_K', 'T_mean_K', 'T_max_K'
      step, dt_step, elapsed_time, interval_time, wall_clock_elapsed, ...
      numel(state.p), min(Tprime), mean(Tprime), max(Tprime)
  };
  writecell(rows, fullfile(step_dir, 'step_info.txt'), 'Delimiter', ',');
end

function local_append_step_history_(out_cfg, state, Tprime, step, dt_step, elapsed_time, wall_clock_elapsed)
  if ~isfield(out_cfg, 'step_history_file') || isempty(out_cfg.step_history_file)
      return;
  end

  fid = fopen(out_cfg.step_history_file, 'a');
  if fid < 0
      error('write_periodic_output_: failed to append step history: %s', out_cfg.step_history_file);
  end

  fprintf(fid, '%d,%.16g,%.16g,%.16g,%.16g,%d,%.16g,%.16g,%.16g\n', ...
          step, dt_step, elapsed_time, out_cfg.interval_time, wall_clock_elapsed, ...
          numel(state.p), min(Tprime), mean(Tprime), max(Tprime));
  fclose(fid);
end

function E = local_particle_energy_vec_(state)
  if isempty(state.p)
      E = zeros(0, 1);
  elseif isfield(state.p, 'E') && ~isempty([state.p.E])
      E = [state.p.E].';
  else
      E = zeros(numel(state.p), 1);
  end
end
