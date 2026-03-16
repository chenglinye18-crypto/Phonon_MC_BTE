function out = run_current_case(run_tag)
% run_current_case Convenience wrapper to run the current input/ case end-to-end.
%
% This wrapper is useful for unattended runs because it:
%   - binds the default input files from input/
%   - disables interactive visualization
%   - writes logs and final summary into the per-run output directory

  if nargin < 1 || isempty(run_tag)
      run_tag = datestr(now, 'yyyymmdd_HHMMSS');
  end

  cs = setup_case_from_ldg_lgrid( ...
      'LdgFile', fullfile('input', 'ldg.txt'), ...
      'LgridFile', fullfile('input', 'lgrid.txt'), ...
      'LengthScale', 1e-6, ...
      'InputLengthUnit', 'um');
  mat = resolve_case_material_(cs);
  opts = mc_default_opts();
  opts.viz.enable = false;
  opts.log.on = true;
  opts.log.to_file = true;
  opts.log.filename = 'mc_log.txt';
  opts.log.print_every = 10;
  opts.output.run_tag = char(run_tag);

  [Tp, p, out] = MC_solve_BTE(cs, mat, opts);
  local_write_final_summary_(out, Tp, p);

  fprintf('FINAL_OK steps=%d converged=%d refreshes=%s Np=%d Tmin=%.6f Tmean=%.6f Tmax=%.6f output=%s\n', ...
          out.nsteps, out.converged, mat2str(out.reservoir_refresh_steps.'), ...
          numel(p), min(Tp), mean(Tp), max(Tp), out.output_dir);
end

function local_write_final_summary_(out, Tp, p)
  if ~isstruct(out) || ~isfield(out, 'output_dir') || isempty(out.output_dir)
      return;
  end

  filepath = fullfile(out.output_dir, 'final_summary.txt');
  fid = fopen(filepath, 'w');
  if fid < 0
      return;
  end

  fprintf(fid, 'steps,%d\n', out.nsteps);
  fprintf(fid, 'converged,%d\n', out.converged);
  if isfield(out, 'reservoir_refresh_steps') && ~isempty(out.reservoir_refresh_steps)
      fprintf(fid, 'reservoir_refresh_steps,"%s"\n', mat2str(out.reservoir_refresh_steps.'));
  else
      fprintf(fid, 'reservoir_refresh_steps,[]\n');
  end
  fprintf(fid, 'Np,%d\n', numel(p));
  fprintf(fid, 'Tmin_K,%.16g\n', min(Tp));
  fprintf(fid, 'Tmean_K,%.16g\n', mean(Tp));
  fprintf(fid, 'Tmax_K,%.16g\n', max(Tp));
  fclose(fid);
end
