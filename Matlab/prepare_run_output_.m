function out_cfg = prepare_run_output_(mesh, opts)
% prepare_run_output_ Create per-run output directory and monitor state.

  cfg = struct();
  if isstruct(opts) && isfield(opts, 'output') && isstruct(opts.output)
      cfg = opts.output;
  end

  out_cfg = struct('enabled', false, ...
                   'every_n_steps', 100, ...
                   'run_dir', '', ...
                   'run_tag', '', ...
                   'inputs_dir', '', ...
                   'steps_dir', '', ...
                   'step_history_file', '', ...
                   'run_wallclock_tic', [], ...
                   'monitors', struct([]), ...
                   'monitor_warnings', {{}}, ...
                   'cum_energy', zeros(0, 1), ...
                   'interval_energy', zeros(0, 1), ...
                   'cum_crossings_pos', zeros(0, 1), ...
                   'cum_crossings_neg', zeros(0, 1), ...
                   'interval_crossings_pos', zeros(0, 1), ...
                   'interval_crossings_neg', zeros(0, 1), ...
                   'cum_time', 0, ...
                   'interval_time', 0);

  out_cfg.enabled = local_get_(cfg, 'enable', false);
  if ~out_cfg.enabled
      return;
  end

  out_cfg.every_n_steps = max(1, round(local_get_(cfg, 'every_n_steps', 100)));
  root_dir = char(local_get_(cfg, 'root_dir', 'output'));
  run_tag = char(local_get_(cfg, 'run_tag', datestr(now, 'yyyymmdd_HHMMSS')));
  run_dir = local_unique_run_dir_(root_dir, run_tag);

  if exist(root_dir, 'dir') ~= 7
      mkdir(root_dir);
  end
  mkdir(run_dir);
  inputs_dir = fullfile(run_dir, 'inputs');
  steps_dir = fullfile(run_dir, 'steps');
  mkdir(inputs_dir);
  mkdir(steps_dir);

  [monitors, warnings] = load_heat_flux_monitors_(mesh, cfg);
  out_cfg.run_dir = run_dir;
  out_cfg.run_tag = run_tag;
  out_cfg.inputs_dir = inputs_dir;
  out_cfg.steps_dir = steps_dir;
  out_cfg.step_history_file = fullfile(run_dir, 'step_history.txt');
  out_cfg.run_wallclock_tic = tic;
  out_cfg.monitors = monitors;
  out_cfg.monitor_warnings = warnings;
  out_cfg.cum_energy = zeros(numel(monitors), 1);
  out_cfg.interval_energy = zeros(numel(monitors), 1);
  out_cfg.cum_crossings_pos = zeros(numel(monitors), 1);
  out_cfg.cum_crossings_neg = zeros(numel(monitors), 1);
  out_cfg.interval_crossings_pos = zeros(numel(monitors), 1);
  out_cfg.interval_crossings_neg = zeros(numel(monitors), 1);

  local_snapshot_inputs_(out_cfg, mesh, opts, cfg);
  local_initialize_step_history_(out_cfg);
  local_write_run_manifest_(out_cfg, cfg);
  local_write_monitor_manifest_(out_cfg);
end

function v = local_get_(s, name, default_v)
  if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
      v = s.(name);
  else
      v = default_v;
  end
end

function run_dir = local_unique_run_dir_(root_dir, run_tag)
  base_dir = fullfile(root_dir, ['run_' run_tag]);
  run_dir = base_dir;
  suffix = 1;
  while exist(run_dir, 'dir') == 7
      run_dir = sprintf('%s_%02d', base_dir, suffix);
      suffix = suffix + 1;
  end
end

function local_snapshot_inputs_(out_cfg, mesh, opts, cfg)
  manifest = {
      'kind', 'source_path', 'snapshot_path'
  };

  file_specs = {
      'layout_ldg', local_get_nested_path_(mesh, {'layout', 'source'})
      'grid_lgrid', local_get_nested_path_(mesh, {'grid', 'source'})
      'initial_temperature', local_get_nested_path_(opts, {'initial_temperature_file'})
      'reference_temperature', local_get_nested_path_(opts, {'reference_temperature_file'})
      'volume_heat_source', local_get_nested_path_(opts, {'volume_heat_source_file'})
      'heat_flux_monitors', local_get_nested_path_(cfg, {'heat_flux_monitor_file'})
  };

  for i = 1:size(file_specs, 1)
      kind = file_specs{i, 1};
      src = file_specs{i, 2};
      if ~ischar(src) && ~isstring(src)
          continue;
      end
      src = char(src);
      if isempty(src) || exist(src, 'file') ~= 2
          continue;
      end

      [~, name, ext] = fileparts(src);
      dst = fullfile(out_cfg.inputs_dir, [kind '__' name ext]);
      copyfile(src, dst);
      manifest(end + 1, :) = {kind, src, dst}; %#ok<AGROW>
  end

  writecell(manifest, fullfile(out_cfg.inputs_dir, 'input_manifest.txt'), 'Delimiter', ',');
end

function value = local_get_nested_path_(s, fields)
  value = '';
  cur = s;
  for i = 1:numel(fields)
      if ~isstruct(cur) || ~isfield(cur, fields{i}) || isempty(cur.(fields{i}))
          return;
      end
      cur = cur.(fields{i});
  end
  value = cur;
end

function local_initialize_step_history_(out_cfg)
  rows = {
      'step', 'dt_s', 'elapsed_time_s', 'interval_time_s', 'wall_clock_elapsed_s', ...
      'Np', 'T_min_K', 'T_mean_K', 'T_max_K'
  };
  writecell(rows, out_cfg.step_history_file, 'Delimiter', ',');
end

function local_write_run_manifest_(out_cfg, cfg)
  rows = {
      'key', 'value'
      'run_tag', out_cfg.run_tag
      'run_dir', out_cfg.run_dir
      'every_n_steps', out_cfg.every_n_steps
      'heat_flux_monitor_file', char(local_get_(cfg, 'heat_flux_monitor_file', ''))
      'monitor_length_scale', local_get_(cfg, 'monitor_length_scale', NaN)
      'created_at', datestr(now, 'yyyy-mm-dd HH:MM:SS')
  };
  writecell(rows, fullfile(out_cfg.run_dir, 'run_manifest.txt'), 'Delimiter', ',');
end

function local_write_monitor_manifest_(out_cfg)
  manifest_path = fullfile(out_cfg.run_dir, 'heat_flux_monitors_manifest.txt');
  rows = {
      'label', 'requested_direction', 'effective_normal', 'area_m2', ...
      'x0_in', 'x1_in', 'y0_in', 'y1_in', 'z0_in', 'z1_in', 'warning'
  };

  for i = 1:numel(out_cfg.monitors)
      m = out_cfg.monitors(i);
      rows(end + 1, :) = {m.label, m.requested_direction, m.effective_normal, m.area, ... %#ok<AGROW>
                          m.bounds_input(1), m.bounds_input(2), ...
                          m.bounds_input(3), m.bounds_input(4), ...
                          m.bounds_input(5), m.bounds_input(6), m.warning};
  end

  writecell(rows, manifest_path, 'Delimiter', ',');

  if ~isempty(out_cfg.monitor_warnings)
      warning_path = fullfile(out_cfg.run_dir, 'heat_flux_monitor_warnings.txt');
      warn_rows = cell(numel(out_cfg.monitor_warnings), 1);
      for i = 1:numel(out_cfg.monitor_warnings)
          warn_rows{i, 1} = out_cfg.monitor_warnings{i};
      end
      writecell(warn_rows, warning_path, 'Delimiter', ',');
  end
end
