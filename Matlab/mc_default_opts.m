function opts = mc_default_opts()
% mc_default_opts Default configuration for the current MC-BTE workflow.
%
% The defaults here are organized around the new file-driven workflow:
%   - geometry / boundary behavior come from ldg+lgrid
%   - initial temperature, reference temperature, heat source, and monitors
%     come from input/ files
%   - output and reservoir refresh are controlled from opts.output / opts.reservoir

% Spectral reference temperature and file-backed per-cell temperature fields.
opts.T0 = [];
opts.initial_temperature_file = fullfile('input', 'initial_temperature.csv');
opts.reference_temperature_file = fullfile('input', 'reference_temperature.txt');

% Time marching controls.
opts.dt = 1e-14;
opts.dt_min = 1e-15;
opts.dt_max = 1e-11;
opts.t_end = 5e-10;
opts.fly_mode = 'cell';

% Dynamic step sizing.
opts.use_dynamic_dt = true;
opts.dt_safety_cfl = 0.5;
opts.p_target = 0.05;

% Monte Carlo packet weights and per-event sampling budgets.
opts.mc_face_particles_per_step = 2e4;
opts.mc_vol_particles_per_step = 2e4;
opts.mc_scatt_particles_per_step = 2e4;
opts.Neff = 1e3;
opts.E_eff = 1e-18;

% Steady-state stopping criteria.
opts.stop_when_steady = true;
opts.steady_tol_inf = 1e-2;
opts.steady_tol_l2 = 1e-2;
opts.steady_min_steps = 50;
opts.steady_min_time = 0.0;
opts.steady_streak_need = 3;

% File-backed volumetric heat source.
opts.volume_heat_source_file = fullfile('input', 'volume_heat_source.txt');

opts.mc_seed = uint64(20240511);
opts.use_common_random_numbers = true;
opts.surface_scatter_range = true;

% Spectral discretization.
opts.n_q = 5000;
opts.n_w = 1000;
opts.weight_by_Cv_for_Q = true;

% Periodic output and monitor settings.
opts.output.enable = true;
opts.output.every_n_steps = 100;
opts.output.root_dir = 'output';
opts.output.run_tag = datestr(now, 'yyyymmdd_HHMMSS');
opts.output.heat_flux_monitor_file = fullfile('input', 'heat_flux_monitors.txt');
opts.output.monitor_length_scale = 1e-6;

% Reservoir refresh settings. Reservoir cells are declared directly in ldg.
opts.reservoir.enable = true;
opts.reservoir.refresh_every_n_steps = 100;
opts.reservoir.refresh_at_step1 = true;

% Scattering-model coefficients.
opts.scatter_on = 1;
opts.PP_BL = 1.18e-24;
opts.PP_BTN = 10.5e-13;
opts.PP_BTU = 2.89e-18;
opts.PP_BLTO = 1 / 3.5e-12;
opts.PB_Tsi = 100e-9;

opts.T_table_min = 1;
opts.T_table_max = 500;
opts.T_table_n = 256;
opts.T_table_log = true;
opts.invert_Newton_iters = 2;

% Global step cap and temperature under-relaxation.
opts.max_steps = 50000;
opts.conv_tol_inf = 1e-4;
opts.conv_tol_l2 = 1e-5;
opts.conv_tol_rel = 1e-6;
opts.conv_n_consec = 3;
opts.T_underrelax = 0.5;

if isempty(gcp('nocreate'))
  parpool('threads');
end
opts.parallel.use_parfor = true;

% Logging and optional visualization hooks.
opts.log.on = true;
opts.log.fly_verbose = true;

opts.viz.enable = true;
opts.viz.out_dir = 'viz_snapshots';
opts.viz.run_tag = datestr(now, 'yyyymmdd_HHMMSS');
opts.viz.every_n = 1;
opts.viz.colormap = 'parula';

opts.deviational = true;
opts.Tref = 350;
opts.use_bin_center_w = true;
opts.mode = 'deviational';

opts.max_points = 5e4;
opts.save_path = 'figs/';
end
