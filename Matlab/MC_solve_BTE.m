function [Tprime, p, out] = MC_solve_BTE(cs, mat, opts)
% MC_solve_BTE Main entry for the ldg/lgrid-driven phonon MC-BTE solver.
%
% Pipeline:
%   1) Build the explicit mesh from the parsed case struct.
%   2) Resolve a scalar linearization temperature T0 when only per-cell fields
%      are supplied by input files.
%   3) Build the spectral lookup tables.
%   4) Initialize deviational particles from T_init and Tref fields.
%   5) March the MC time loop until convergence or max_steps.

if nargin < 3 || isempty(opts)
    opts = mc_default_opts();
end

rng(opts.mc_seed, 'twister');

mesh = init_mesh_from_geom_(cs);
if isstruct(mat) && isfield(mat, 'material_library') && ~isempty(mat.material_library)
    mesh.material_library = mat.material_library;
end
opts = resolve_linearization_temperature_(mesh, opts);
spec = build_spectral_grid_(mat, opts);

state = init_state_Energy(mesh, spec, opts);
[Tprime, p, out] = MC_time_loop_BTE(mesh, spec, opts, state);
end

function opts = resolve_linearization_temperature_(mesh, opts)
% resolve_linearization_temperature_ Pick a scalar T0 for spectral linearization.
%
% T0 is only used for lookup/spectral tabulation. The actual initial field still
% comes from the per-cell temperature file.
if isfield(opts, 'T0') && ~isempty(opts.T0) && isfinite(opts.T0)
    return;
end

fallback_T = NaN;
if isfield(opts, 'Tref') && isfinite(opts.Tref)
    fallback_T = opts.Tref;
elseif isfield(opts, 'T_init') && isfinite(opts.T_init)
    fallback_T = opts.T_init;
end

[~, meta] = load_initial_temperature_field_(mesh, opts, fallback_T);
if isfinite(meta.T_mean)
    opts.T0 = meta.T_mean;
elseif isfinite(fallback_T)
    opts.T0 = fallback_T;
else
    opts.T0 = 300;
end
end
