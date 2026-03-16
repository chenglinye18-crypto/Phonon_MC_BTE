function out_tbl = export_thermal_conductivity_csv(run_dir, steps, output_csv)
% export_thermal_conductivity_csv Export interval-based thermal conductivity to CSV.
%
% Usage:
%   export_thermal_conductivity_csv(fullfile('output','run_20260315_122830'), ...
%       [100 200 300], fullfile('output','run_20260315_122830','thermal_k.csv'));
%
% Inputs:
%   run_dir    : run output folder, e.g. output/run_20260315_122830
%   steps      : numeric step list, or 'all' / [] for all available steps
%   output_csv : destination CSV file; if omitted a default file is created in run_dir
%
% Method:
%   - heat flux uses flux_interval_W_m2 from each step's heat_flux.txt
%   - temperatures use the cells immediately on the two sides of each monitor plane
%   - deltaL uses the difference between the weighted mean center positions on both sides
%   - conductivity uses the user-requested formula:
%         k = q / (deltaT / deltaL)

  if nargin < 1 || isempty(run_dir)
      error('export_thermal_conductivity_csv: run_dir is required.');
  end
  if nargin < 2 || isempty(steps)
      steps = 'all';
  end

  root_dir = fileparts(fileparts(mfilename('fullpath')));
  addpath(root_dir);

  run_dir = local_resolve_path_(run_dir, root_dir);
  if exist(run_dir, 'dir') ~= 7
      error('export_thermal_conductivity_csv: run folder not found: %s', run_dir);
  end

  if nargin < 3 || isempty(output_csv)
      output_csv = fullfile(run_dir, local_default_output_name_(steps));
  else
      output_csv = local_resolve_path_(output_csv, root_dir);
  end
  local_ensure_parent_dir_(output_csv);

  [ldg_file, lgrid_file] = local_find_run_input_snapshots_(run_dir);
  cs = setup_case_from_ldg_lgrid( ...
      'LdgFile', ldg_file, ...
      'LgridFile', lgrid_file, ...
      'LengthScale', 1e-6, ...
      'InputLengthUnit', 'um', ...
      'Verbose', false);
  mesh = init_mesh_from_geom_(cs);

  monitor_tbl = readtable(fullfile(run_dir, 'heat_flux_monitors_manifest.txt'), ...
                          'Delimiter', ',', 'VariableNamingRule', 'preserve');
  step_list = local_resolve_steps_(run_dir, steps);

  rows = repmat(local_empty_row_(), 0, 1);
  for is = 1:numel(step_list)
      step = step_list(is);
      step_dir = fullfile(run_dir, 'steps', sprintf('step_%05d', step));
      temp_tbl = readtable(fullfile(step_dir, 'temperature.txt'), ...
                           'Delimiter', ',', 'VariableNamingRule', 'preserve');
      heat_tbl = readtable(fullfile(step_dir, 'heat_flux.txt'), ...
                           'Delimiter', ',', 'VariableNamingRule', 'preserve');
      info_tbl = readtable(fullfile(step_dir, 'step_info.txt'), ...
                           'Delimiter', ',', 'VariableNamingRule', 'preserve');

      Tcell = local_temperature_vector_(mesh, temp_tbl);
      step_info = info_tbl(1, :);

      for im = 1:height(monitor_tbl)
          mon = monitor_tbl(im, :);
          heat_row = heat_tbl(strcmp(string(heat_tbl.label), string(mon.label)), :);
          if isempty(heat_row)
              continue;
          end

          side = local_monitor_side_cells_(mesh, mon);
          T_minus = local_weighted_average_(Tcell(side.minus_cells), side.minus_weights);
          T_plus = local_weighted_average_(Tcell(side.plus_cells), side.plus_weights);
          x_minus = local_weighted_average_(side.minus_centers, side.minus_weights);
          x_plus = local_weighted_average_(side.plus_centers, side.plus_weights);

          deltaT = T_plus - T_minus;
          deltaL = x_plus - x_minus;
          gradT = deltaT / deltaL;
          q_flux = heat_row.flux_interval_W_m2(1);
          k_value = q_flux / gradT;

          row = local_empty_row_();
          row.run_dir = string(run_dir);
          row.step = step;
          row.label = string(mon.label{1});
          row.requested_direction = string(mon.requested_direction{1});
          row.effective_normal = string(mon.effective_normal{1});
          row.area_m2 = mon.area_m2(1);
          row.elapsed_time_s = step_info.elapsed_time_s(1);
          row.interval_time_s = step_info.interval_time_s(1);
          row.interval_energy_net_J = heat_row.interval_energy_net_J(1);
          row.q_flux_interval_W_m2 = q_flux;
          row.minus_cell_ids = string(local_format_cell_ids_(mesh, side.minus_cells));
          row.plus_cell_ids = string(local_format_cell_ids_(mesh, side.plus_cells));
          row.minus_cell_count = numel(side.minus_cells);
          row.plus_cell_count = numel(side.plus_cells);
          row.minus_center_m = x_minus;
          row.plus_center_m = x_plus;
          row.deltaL_m = deltaL;
          row.T_minus_K = T_minus;
          row.T_plus_K = T_plus;
          row.deltaT_K = deltaT;
          row.gradT_K_per_m = gradT;
          row.k_W_m_K = k_value;
          rows(end + 1, 1) = row; %#ok<AGROW>
      end
  end

  out_tbl = struct2table(rows);
  writetable(out_tbl, output_csv);
  fprintf('export_thermal_conductivity_csv: wrote %d rows to %s\n', height(out_tbl), output_csv);
end

function path_out = local_resolve_path_(path_in, root_dir)
  path_in = char(string(path_in));
  if isempty(path_in)
      path_out = path_in;
      return;
  end

  if ~isempty(regexp(path_in, '^[A-Za-z]:[\\/]', 'once'))
      path_out = path_in;
  elseif startsWith(path_in, filesep)
      path_out = path_in;
  else
      if exist(path_in, 'dir') == 7 || exist(path_in, 'file') == 2
          path_out = path_in;
      else
          path_out = fullfile(root_dir, path_in);
      end
  end
end

function filename = local_default_output_name_(steps)
  if isnumeric(steps) && ~isempty(steps)
      filename = sprintf('thermal_conductivity_steps_%05d_%05d.csv', min(steps), max(steps));
  else
      filename = 'thermal_conductivity_all_steps.csv';
  end
end

function [ldg_file, lgrid_file] = local_find_run_input_snapshots_(run_dir)
  inputs_dir = fullfile(run_dir, 'inputs');
  ldg_match = dir(fullfile(inputs_dir, 'layout_ldg__*.txt'));
  lgrid_match = dir(fullfile(inputs_dir, 'grid_lgrid__*.txt'));
  if isempty(ldg_match) || isempty(lgrid_match)
      error('export_thermal_conductivity_csv: missing input snapshots under %s', inputs_dir);
  end
  ldg_file = fullfile(inputs_dir, ldg_match(1).name);
  lgrid_file = fullfile(inputs_dir, lgrid_match(1).name);
end

function step_list = local_resolve_steps_(run_dir, steps)
  dirs = dir(fullfile(run_dir, 'steps', 'step_*'));
  available = zeros(numel(dirs), 1);
  for i = 1:numel(dirs)
      tok = regexp(dirs(i).name, '^step_(\d+)$', 'tokens', 'once');
      if ~isempty(tok)
          available(i) = str2double(tok{1});
      end
  end
  available = sort(available(available > 0));

  if (ischar(steps) || isstring(steps)) && strcmpi(char(string(steps)), 'all')
      step_list = available;
      return;
  end

  step_list = unique(round(steps(:)));
  missing = setdiff(step_list, available);
  if ~isempty(missing)
      error('export_thermal_conductivity_csv: requested steps not found: %s', mat2str(missing.'));
  end
end

function Tcell = local_temperature_vector_(mesh, temp_tbl)
  lin = sub2ind([mesh.Nx, mesh.Ny, mesh.Nz], ...
                temp_tbl.idxcell, temp_tbl.idycell, temp_tbl.idzcell);
  Tcell = zeros(mesh.Nc, 1);
  Tcell(lin) = temp_tbl.Temperature;
end

function side = local_monitor_side_cells_(mesh, mon)
  normal = char(mon.effective_normal{1});
  b = [mon.x0_in(1), mon.x1_in(1), mon.y0_in(1), mon.y1_in(1), mon.z0_in(1), mon.z1_in(1)] * 1e-6;
  boxes = mesh.boxes;
  centers = mesh.centers;
  tol = 1e-12 * max(1, max(abs([boxes(:); b(:)])));

  switch upper(normal(end))
    case 'X'
      coord = 0.5 * (b(1) + b(2));
      minus_touch = abs(boxes(:, 2) - coord) <= tol;
      plus_touch = abs(boxes(:, 1) - coord) <= tol;
      tangential = local_overlap_(boxes(:, 3), boxes(:, 4), b(3), b(4), tol) & ...
                   local_overlap_(boxes(:, 5), boxes(:, 6), b(5), b(6), tol);
      minus_weights = local_overlap_length_(boxes(:, 3), boxes(:, 4), b(3), b(4)) .* ...
                      local_overlap_length_(boxes(:, 5), boxes(:, 6), b(5), b(6));
      plus_weights = minus_weights;
      minus_centers = centers(:, 1);
      plus_centers = centers(:, 1);

    case 'Y'
      coord = 0.5 * (b(3) + b(4));
      minus_touch = abs(boxes(:, 4) - coord) <= tol;
      plus_touch = abs(boxes(:, 3) - coord) <= tol;
      tangential = local_overlap_(boxes(:, 1), boxes(:, 2), b(1), b(2), tol) & ...
                   local_overlap_(boxes(:, 5), boxes(:, 6), b(5), b(6), tol);
      minus_weights = local_overlap_length_(boxes(:, 1), boxes(:, 2), b(1), b(2)) .* ...
                      local_overlap_length_(boxes(:, 5), boxes(:, 6), b(5), b(6));
      plus_weights = minus_weights;
      minus_centers = centers(:, 2);
      plus_centers = centers(:, 2);

    case 'Z'
      coord = 0.5 * (b(5) + b(6));
      minus_touch = abs(boxes(:, 6) - coord) <= tol;
      plus_touch = abs(boxes(:, 5) - coord) <= tol;
      tangential = local_overlap_(boxes(:, 1), boxes(:, 2), b(1), b(2), tol) & ...
                   local_overlap_(boxes(:, 3), boxes(:, 4), b(3), b(4), tol);
      minus_weights = local_overlap_length_(boxes(:, 1), boxes(:, 2), b(1), b(2)) .* ...
                      local_overlap_length_(boxes(:, 3), boxes(:, 4), b(3), b(4));
      plus_weights = minus_weights;
      minus_centers = centers(:, 3);
      plus_centers = centers(:, 3);

    otherwise
      error('export_thermal_conductivity_csv: unsupported monitor normal %s', normal);
  end

  minus_idx = find(minus_touch & tangential);
  plus_idx = find(plus_touch & tangential);
  if isempty(minus_idx) || isempty(plus_idx)
      error('export_thermal_conductivity_csv: failed to locate cells adjacent to monitor %s', mon.label{1});
  end

  side = struct();
  side.minus_cells = minus_idx(:);
  side.plus_cells = plus_idx(:);
  side.minus_weights = minus_weights(minus_idx);
  side.plus_weights = plus_weights(plus_idx);
  side.minus_centers = minus_centers(minus_idx);
  side.plus_centers = plus_centers(plus_idx);
end

function tf = local_overlap_(a0, a1, b0, b1, tol)
  tf = (a1 > b0 + tol) & (a0 < b1 - tol | abs(a0 - b1) <= tol | abs(a1 - b0) <= tol);
end

function len = local_overlap_length_(a0, a1, b0, b1)
  len = max(0, min(a1, b1) - max(a0, b0));
end

function v = local_weighted_average_(values, weights)
  values = values(:);
  weights = weights(:);
  if isempty(values)
      v = NaN;
      return;
  end
  if isempty(weights) || all(weights <= 0)
      v = mean(values);
      return;
  end
  v = sum(values .* weights) / sum(weights);
end

function txt = local_format_cell_ids_(mesh, cell_ids)
  parts = strings(numel(cell_ids), 1);
  for i = 1:numel(cell_ids)
      [ix, iy, iz] = ind2sub([mesh.Nx, mesh.Ny, mesh.Nz], cell_ids(i));
      parts(i) = sprintf('%d:%d:%d', ix, iy, iz);
  end
  txt = strjoin(parts, ';');
end

function row = local_empty_row_()
  row = struct('run_dir', "", ...
               'step', 0, ...
               'label', "", ...
               'requested_direction', "", ...
               'effective_normal', "", ...
               'area_m2', NaN, ...
               'elapsed_time_s', NaN, ...
               'interval_time_s', NaN, ...
               'interval_energy_net_J', NaN, ...
               'q_flux_interval_W_m2', NaN, ...
               'minus_cell_ids', "", ...
               'plus_cell_ids', "", ...
               'minus_cell_count', 0, ...
               'plus_cell_count', 0, ...
               'minus_center_m', NaN, ...
               'plus_center_m', NaN, ...
               'deltaL_m', NaN, ...
               'T_minus_K', NaN, ...
               'T_plus_K', NaN, ...
               'deltaT_K', NaN, ...
               'gradT_K_per_m', NaN, ...
               'k_W_m_K', NaN);
end

function local_ensure_parent_dir_(filepath)
  parent_dir = fileparts(filepath);
  if ~isempty(parent_dir) && exist(parent_dir, 'dir') ~= 7
      mkdir(parent_dir);
  end
end
