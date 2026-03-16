clc;
clear all;

% main.m
% Top-level run script for the current ldg/lgrid-driven phonon MC-BTE case.
% Flow:
%   1) Read geometry, mesh, regions, face rules, and reservoirs from input/.
%   2) Resolve the material model(s) referenced by the layout.
%   3) Build default solver options and launch the time-domain MC solver.

fprintf('*****************************************************\n');
fprintf('*          BTE Monte Carlo Simulator V1.0           *\n');
fprintf('*    Developed by Chenglin Ye, Peking University    *\n');
fprintf('*           Release Date: 20th Oct, 2025            *\n');
fprintf('*****************************************************\n');

% Geometry and boundary behavior are now fully file-driven.
cs = setup_case_from_ldg_lgrid( ...
    'LdgFile', fullfile('input', 'ldg.txt'), ...
    'LgridFile', fullfile('input', 'lgrid.txt'), ...
    'LengthScale', 1e-6, ...
    'InputLengthUnit', 'um');

% Material resolution stays separate from geometry so multi-material extension
% can keep using the same geometry loader.
mat = resolve_case_material_(cs);
opts = mc_default_opts();

[Tp, p, out] = MC_solve_BTE(cs, mat, opts);
