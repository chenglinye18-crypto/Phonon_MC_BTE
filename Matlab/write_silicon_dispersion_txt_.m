function write_silicon_dispersion_txt_(varargin)
% write_silicon_dispersion_txt_ Generate Silicon dispersion text data.
%
% Output columns:
%   branch_id, q_real(1/m), f(THz), vg(m/s)

  parser = inputParser;
  parser.FunctionName = mfilename;
  addParameter(parser, 'FilePath', fullfile('input', 'phonon_dispersion_Si.txt'), ...
      @(x) ischar(x) || isstring(x));
  addParameter(parser, 'NumQ', 5000, ...
      @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 2 && mod(x, 1) == 0);
  parse(parser, varargin{:});

  filepath = char(parser.Results.FilePath);
  nq_tab = double(parser.Results.NumQ);

  a0 = 5.431e-10;
  qmax = 2 * pi / a0;
  q = linspace(0, qmax, nq_tab + 1);

  omega0 = [0.00, 0.00, 9.88, 10.20] * 1e13;
  vs = [9.01, 5.23, 0.00, -2.57] * 1e3;
  cpar = [-2.00, -2.26, -1.60, 1.11] * 1e-7;

  fid = fopen(filepath, 'w');
  if fid < 0
      error('write_silicon_dispersion_txt_: failed to open %s for writing.', filepath);
  end

  cleaner = onCleanup(@() fclose(fid));
  fprintf(fid, '# branch\tq_real(1/m)\tf(THz)\tvg(m/s)\n');

  for b = 1:numel(omega0)
      omega = max(omega0(b) + vs(b) .* q + cpar(b) .* (q .^ 2), 0);
      f_thz = omega / (2 * pi * 1e12);
      vg = vs(b) + 2 * cpar(b) .* q;

      for iq = 1:numel(q)
          fprintf(fid, '%d\t%.10e\t%.10f\t%.10f\n', b, q(iq), f_thz(iq), vg(iq));
      end
  end
end
