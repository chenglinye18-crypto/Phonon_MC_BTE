function [Tprime, p, out] = MC_time_loop_BTE(mesh, spec, opts, state)
% MC_time_loop_BTE Advance the rule-driven MC-BTE system in time.
%
% Main loop order per step:
%   1) Optionally refresh reservoir cells to their equilibrium distributions.
%   2) Spawn volumetric heat-source packets.
%   3) Fly particles with ldg-driven face actions.
%   4) Scatter particles.
%   5) Recover cell temperature from particle energy.
%   6) Write periodic outputs and check steady-state convergence.

    Nc = infer_Nc_(mesh);
    T0 = get_or(opts, 'T0', 300);
    dt_min = get_or(opts, 'dt_min', 1e-15);
    dt_max = get_or(opts, 'dt_max', 1e-10);
    max_steps = get_or(opts, 'max_steps', 5000);
    alpha_T = max(min(get_or(opts, 'T_underrelax', 1.0), 1), 0);
    scatter_on = get_or(opts, 'scatter_on', true);
    reservoir_cfg = getfield_or(opts, 'reservoir', struct());
    reservoir_on = get_or(reservoir_cfg, 'enable', true);
    reservoir_refresh_every = max(1, round(get_or(reservoir_cfg, 'refresh_every_n_steps', 100)));
    reservoir_refresh_at_step1 = get_or(reservoir_cfg, 'refresh_at_step1', true);

    [qvol, qsrc_meta] = load_volume_heat_source_field_(mesh, opts, 0);
    use_volume_map = any(qvol ~= 0);

    output_cfg = prepare_run_output_(mesh, opts);
    if output_cfg.enabled
        mesh.heat_flux_monitors = output_cfg.monitors;
        if isfield(opts, 'viz') && isstruct(opts.viz)
            opts.viz.out_dir = fullfile(output_cfg.run_dir, 'viz_snapshots');
        end
        opts.save_path = fullfile(output_cfg.run_dir, 'figs');
    end

    logcfg = get_or(opts, 'log', struct());
    log_on = get_or(logcfg, 'on', true);
    print_every = get_or(logcfg, 'print_every', 1);
    to_file = get_or(logcfg, 'to_file', false);
    logfile = get_or(logcfg, 'filename', 'mc_log.txt');
    if to_file && output_cfg.enabled
        [~, logname, logext] = fileparts(logfile);
        if isempty(logext), logext = '.txt'; end
        logfile = fullfile(output_cfg.run_dir, [logname logext]);
    end
    if to_file
        fid = fopen(logfile, 'a');
        if fid < 0, fid = 1; end
    else
        fid = 1;
    end

    stop_when_steady = get_or(opts, 'stop_when_steady', true);
    conv_cfg = getfield_or(opts, 'conv', struct());
    conv = struct();
    conv.enabled = get_or(conv_cfg, 'enable', stop_when_steady);
    conv.min_steps = max(1, round(get_or(conv_cfg, 'min_steps', get_or(opts, 'steady_min_steps', max_steps))));
    conv.n_consec = max(1, round(get_or(conv_cfg, 'n_consec', get_or(opts, 'steady_streak_need', 3))));
    conv.tol_inf = get_or(conv_cfg, 'tol_inf', get_or(opts, 'steady_tol_inf', 5e-2));
    conv.tol_l2 = get_or(conv_cfg, 'tol_l2', get_or(opts, 'steady_tol_l2', 5e-2));
    conv.tol_Enet = get_or(conv_cfg, 'tol_Enet', 2e-18);

    consec_ok = 0;

    hdr = [ ...
      '  step |      dt[s]   |  dT_inf[K] |  dT_L2[K] |    E_net[J] |' ...
      '   T_min[K] |  T_mean[K] |   T_max[K] |  pscat_max |      Np' newline];
    fmt = '%6d | %1.4e | %1.4e | %1.4e | %+.3e | %9.3f | %10.3f | %9.3f | %10.3f | %7d\n';

    [Tstar, Tinit_meta] = initial_temperature_from_state_or_file_(state, mesh, opts, T0);
    Tprime = Tstar;

    out = struct('dt_hist', [], 'T_inf_hist', [], 'T_l2_hist', [], 'pscat_max_hist', [], ...
                 'E_net_hist', [], 'dU_cells_hist', [], 'dU_alive_hist', [], 'resid_hist', [], ...
                 'iface_hist', [], 'nsteps', 0, 'converged', false, 'Temperature_hist', [], ...
                 'initial_temperature', Tstar, 'initial_temperature_meta', Tinit_meta, ...
                 'output_dir', '', 'output_steps_dir', '', 'step_history_file', '', ...
                 'heat_flux_monitor_warnings', {{}}, 'reservoir_refresh_steps', []);

    LUT = build_E_T_lookup_(spec, struct('T_min', 1, 'T_max', 2000, 'nT', 2001));
    U_of_T = @(T) interp1(LUT.T, LUT.U, clamp_vec(T, LUT.T(1), LUT.T(end)), 'pchip');
    Vc = cell_volumes_(mesh);
    active_mask = active_cell_mask_(mesh);

    if reservoir_on && local_has_reservoirs_(mesh) && reservoir_refresh_at_step1
        [state, res_info] = refresh_reservoir_particles_(state, mesh, spec, opts);
        if res_info.refreshed
            Tstar(res_info.cell_ids) = res_info.target_temperature_cell;
            Tprime(res_info.cell_ids) = res_info.target_temperature_cell;
            out.reservoir_refresh_steps(end + 1, 1) = 1;
            if log_on
                fprintf(fid, '[reservoir] step=%d refreshed %d cells | removed=%d added=%d\n', ...
                        1, numel(res_info.cell_ids), res_info.removed_particles, res_info.added_particles);
            end
        end
    end

    U_alive_prev = particles_total_energy_(state, opts);
    U_cells_prev = sum(U_of_T(Tstar) .* Vc);

    if log_on
        fprintf(fid, '[%s] MC BTE start. Ncells=%d, T0=%.2f K, Tinit=[%.2f, %.2f, %.2f] K\n', ...
                datestr(now, 'HH:MM:SS'), Nc, T0, min(Tstar), mean(Tstar), max(Tstar));
        if qsrc_meta.used_file
            fprintf(fid, '[source] loaded volume heat source from %s | q[min,mean,max]=[%+.3e, %+.3e, %+.3e]\n', ...
                    qsrc_meta.source, qsrc_meta.q_min, qsrc_meta.q_mean, qsrc_meta.q_max);
        end
        fprintf(fid, hdr);
    end

    for step = 1:max_steps
        % draw_iter_snapshot(state, spec, mesh, step, opts);

        if reservoir_on && local_has_reservoirs_(mesh) && ...
                local_should_refresh_reservoirs_(step, reservoir_refresh_every, reservoir_refresh_at_step1)
            [state, res_info] = refresh_reservoir_particles_(state, mesh, spec, opts);
            if res_info.refreshed
                Tstar(res_info.cell_ids) = res_info.target_temperature_cell;
                Tprime(res_info.cell_ids) = res_info.target_temperature_cell;
                out.reservoir_refresh_steps(end + 1, 1) = step;
                if log_on
                    fprintf(fid, '[reservoir] step=%d refreshed %d cells | removed=%d added=%d\n', ...
                            step, numel(res_info.cell_ids), res_info.removed_particles, res_info.added_particles);
                end
            end
        end

        if scatter_on
            r_tau = precompute_relax_times_(state, Tstar, opts, spec);
        else
            r_tau = [];
        end

        [dt, info_dt] = step_pick_dt_(mesh, spec, opts, state, r_tau);
        if ~isfinite(dt), dt = dt_min; end
        dt = min(max(dt, dt_min), dt_max);
        out.dt_hist(end + 1, 1) = dt;

        if use_volume_map
            newpV = spawn_volume_sources_from_map_(qvol, opts, mesh, spec, state, Tprime, LUT, dt);
            if ~isempty(newpV)
                state.p = [state.p; newpV];
            end
        end

        [state, fly_stats] = particle_fly_(state, mesh, dt, opts, spec);
        output_cfg = local_accumulate_output_(output_cfg, fly_stats, dt);

        if scatter_on
            r_tau = precompute_relax_times_(state, Tstar, opts, spec);
            try
                [state, ~] = particle_scattering_(state, mesh, spec, opts, dt, Tstar, r_tau);
            catch
                state = particle_scattering_(state, mesh, spec, opts, dt, Tstar, r_tau);
            end
        end

        [Tnew, ~] = update_temperature_from_energy_(state, mesh, spec, opts, LUT);
        Tprime = Tnew;

        E_net_total = 0;
        out.E_net_hist(end + 1, 1) = E_net_total;
        out.iface_hist{end + 1} = struct();
        out.Temperature_hist(end + 1, :) = Tprime;

        U_alive_now = particles_total_energy_(state, opts);
        U_cells_now = sum(U_of_T(Tprime) .* Vc);

        dU_cells = U_cells_now - U_cells_prev;
        dU_alive = U_alive_now - U_alive_prev;
        resid = dU_cells - dU_alive;

        out.dU_cells_hist(end + 1, 1) = dU_cells;
        out.dU_alive_hist(end + 1, 1) = dU_alive;
        out.resid_hist(end + 1, 1) = resid;

        dT = Tprime(active_mask) - Tstar(active_mask);
        T_inf = norm(dT, inf);
        T_l2 = norm(dT) / sqrt(max(nnz(active_mask), 1));
        out.T_inf_hist(end + 1, 1) = T_inf;
        out.T_l2_hist(end + 1, 1) = T_l2;

        pscat_max = NaN;
        if isstruct(info_dt) && isfield(info_dt, 'p_scat_max') && ~isempty(info_dt.p_scat_max)
            pscat_max = info_dt.p_scat_max;
        end
        out.pscat_max_hist(end + 1, 1) = pscat_max;

        if log_on && (step == 1 || mod(step, print_every) == 0)
            fprintf(fid, hdr);
            Tmin = min(Tprime);
            Tmean = mean(Tprime);
            Tmax = max(Tprime);
            fprintf(fid, fmt, step, dt, T_inf, T_l2, E_net_total, Tmin, Tmean, Tmax, ...
                    safe_num(pscat_max, 0), numel(state.p));

            fprintf(fid, '   [ledger] dUc=%+.3e  dUa=%+.3e  resid=%+.3e  (J)\n', dU_cells, dU_alive, resid);
            if isstruct(info_dt) && (isfield(info_dt, 'dt_cfl') || isfield(info_dt, 'dt_prob'))
                fprintf(fid, '   [dt] dt_cfl=%1.3e  dt_prob=%1.3e\n', ...
                    safe_num(getfield_or(info_dt, 'dt_cfl', NaN), NaN), ...
                    safe_num(getfield_or(info_dt, 'dt_prob', NaN), NaN));
            end

            if to_file && fid > 1
                fclose(fid);
                fid = fopen(logfile, 'a');
                if fid < 0, fid = 1; end
            end
            drawnow limitrate;
        end

        if output_cfg.enabled && mod(step, output_cfg.every_n_steps) == 0
            write_periodic_output_(output_cfg, mesh, spec, state, Tprime, step, output_cfg.cum_time, dt);
            output_cfg = local_reset_output_interval_(output_cfg);
        end

        if alpha_T < 1
            Tstar = (1 - alpha_T) * Tstar + alpha_T * Tprime;
        else
            Tstar = Tprime;
        end
        U_cells_prev = U_cells_now;
        U_alive_prev = U_alive_now;
        out.nsteps = step;

        pass_now = (T_inf <= conv.tol_inf) && (T_l2 <= conv.tol_l2) && (abs(E_net_total) <= conv.tol_Enet);
        if pass_now
            consec_ok = consec_ok + 1;
        else
            consec_ok = 0;
        end

        if conv.enabled && (step >= conv.min_steps) && (consec_ok >= conv.n_consec)
            out.converged = true;
            out.nsteps = step;
            if log_on
                fprintf(fid, '[%s] Converged at step %d: dT_inf=%1.3e, dT_L2=%1.3e, E_net=%+.3e\n', ...
                        datestr(now, 'HH:MM:SS'), step, T_inf, T_l2, E_net_total);
            end
            break;
        end
    end

    if output_cfg.enabled
        if output_cfg.interval_time > 0 && out.nsteps > 0
            write_periodic_output_(output_cfg, mesh, spec, state, Tprime, out.nsteps, output_cfg.cum_time, out.dt_hist(end));
        end
        out.output_dir = output_cfg.run_dir;
        out.output_steps_dir = output_cfg.steps_dir;
        out.step_history_file = output_cfg.step_history_file;
        out.heat_flux_monitor_warnings = output_cfg.monitor_warnings;
    end

    if log_on && to_file && fid > 1, fclose(fid); end
    p = state.p;
end

function Nc = infer_Nc_(mesh)
    if isfield(mesh, 'Nc') && ~isempty(mesh.Nc), Nc = mesh.Nc; return; end
    if all(isfield(mesh, {'Nx', 'Ny', 'Nz'})), Nc = mesh.Nx * mesh.Ny * mesh.Nz; return; end
    if isfield(mesh, 'boxes'), Nc = size(mesh.boxes, 1); return; end
    error('infer_Nc_: cannot infer Nc from mesh.');
end

function v = get_or(s, name, default_v)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default_v; end
end

function v = getfield_or(s, name, default_v)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default_v; end
end

function x = clamp_vec(x, a, b)
    x = min(max(x, a), b);
end

function Vc = cell_volumes_(mesh)
    if all(isfield(mesh, {'Nx', 'Ny', 'Nz', 'x_edges', 'y_edges', 'z_edges'}))
        dx = diff(mesh.x_edges(:)); dy = diff(mesh.y_edges(:)); dz = diff(mesh.z_edges(:));
        Vc3 = reshape(dx, [], 1) .* reshape(dy, 1, []) .* reshape(dz, 1, 1, []);
        Vc = Vc3(:);
    elseif isfield(mesh, 'boxes') && ~isempty(mesh.boxes)
        bx = mesh.boxes;
        Vc = (bx(:, 2) - bx(:, 1)) .* (bx(:, 4) - bx(:, 3)) .* (bx(:, 6) - bx(:, 5));
    elseif isfield(mesh, 'cell_vol') && ~isempty(mesh.cell_vol)
        Vc = mesh.cell_vol(:);
    else
        error('cell_volumes_: mesh lacks edges/boxes/cell_vol.');
    end
end

function E = particles_total_energy_(state, opts)
    if isempty(state.p), E = 0; return; end
    if isfield(state.p, 'E') && ~isempty([state.p.E])
        E = sum([state.p.E]);
    else
        E_eff = get_or(opts, 'E_eff', 1e-18);
        E = E_eff * numel(state.p);
    end
end

function v = safe_num(v, fallback)
    if ~isfinite(v), v = fallback; end
end

function newp = spawn_volume_sources_from_map_(qvol, opts, mesh, spec, state, Tprime, LUT, dt)
    if isempty(qvol), newp = struct([]); return; end
    E_eff = state.WE;
    force_pos = true;
    newp = struct([]);
    for cid = 1:numel(qvol)
        q = qvol(cid);
        if q == 0, continue; end
        src = struct('type', 'volume', ...
                     'qvol', q, ...
                     'E_eff', E_eff, ...
                     'region', struct('type', 'cells', 'id', cid), ...
                     'force_positive', force_pos);
        chunk = spawn_heat_source(opts, mesh, spec, state, Tprime, LUT, src, dt);
        if ~isempty(chunk)
            if isempty(newp)
                newp = chunk;
            else
                newp = [newp; chunk]; %#ok<AGROW>
            end
        end
    end
end

function mask = active_cell_mask_(mesh)
    mask = true(infer_Nc_(mesh), 1);
end

function [Tcell, meta] = initial_temperature_from_state_or_file_(state, mesh, opts, default_T)
    meta = struct('source', '', 'used_file', false, 'T_min', NaN, 'T_mean', NaN, 'T_max', NaN);

    if isstruct(state) && isfield(state, 'info') && isstruct(state.info) ...
            && isfield(state.info, 'T_init_cell') && numel(state.info.T_init_cell) == infer_Nc_(mesh)
        Tcell = state.info.T_init_cell(:);
        if isfield(state.info, 'initial_temperature_meta') && isstruct(state.info.initial_temperature_meta)
            meta = state.info.initial_temperature_meta;
        end
        meta.T_min = min(Tcell);
        meta.T_mean = mean(Tcell);
        meta.T_max = max(Tcell);
        return;
    end

    [Tcell, meta] = load_initial_temperature_field_(mesh, opts, default_T);
end

function output_cfg = local_accumulate_output_(output_cfg, fly_stats, dt)
    if ~isstruct(output_cfg) || ~isfield(output_cfg, 'enabled') || ~output_cfg.enabled
        return;
    end

    output_cfg.cum_time = output_cfg.cum_time + dt;
    output_cfg.interval_time = output_cfg.interval_time + dt;

    if isstruct(fly_stats) && isfield(fly_stats, 'heat_flux') && ~isempty(fly_stats.heat_flux)
        h = fly_stats.heat_flux;
        if isfield(h, 'net_energy') && ~isempty(h.net_energy)
            output_cfg.cum_energy = output_cfg.cum_energy + h.net_energy;
            output_cfg.interval_energy = output_cfg.interval_energy + h.net_energy;
            output_cfg.cum_crossings_pos = output_cfg.cum_crossings_pos + h.crossings_pos;
            output_cfg.cum_crossings_neg = output_cfg.cum_crossings_neg + h.crossings_neg;
            output_cfg.interval_crossings_pos = output_cfg.interval_crossings_pos + h.crossings_pos;
            output_cfg.interval_crossings_neg = output_cfg.interval_crossings_neg + h.crossings_neg;
        end
    end
end

function output_cfg = local_reset_output_interval_(output_cfg)
    output_cfg.interval_energy(:) = 0;
    output_cfg.interval_crossings_pos(:) = 0;
    output_cfg.interval_crossings_neg(:) = 0;
    output_cfg.interval_time = 0;
end

function tf = local_has_reservoirs_(mesh)
    tf = isfield(mesh, 'reservoirs') && ~isempty(mesh.reservoirs);
end

function tf = local_should_refresh_reservoirs_(step, every_n_steps, refresh_at_step1)
    %#ok<INUSD>
    tf = (step > 1) && (mod(step - 1, every_n_steps) == 0);
end
