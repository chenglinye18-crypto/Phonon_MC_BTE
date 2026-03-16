function si = mat_silicon_100(varargin)
% mat_silicon_100 Load Silicon (100) phonon dispersion from text data.
%
% Default source file:
%   input/phonon_dispersion_Si.txt

  parser = inputParser;
  parser.FunctionName = mfilename;
  addParameter(parser, 'FilePath', fullfile('input', 'phonon_dispersion_Si.txt'), ...
      @(x) ischar(x) || isstring(x));
  addParameter(parser, 'BranchNames', {'LA', 'TA', 'LO', 'TO'}, @(x) iscell(x) || isstring(x));
  addParameter(parser, 'Degeneracy', [1, 2, 1, 2], @isnumeric);
  parse(parser, varargin{:});

  si = mat_from_phonon_dispersion_file_( ...
      'FilePath', parser.Results.FilePath, ...
      'MaterialName', 'Silicon (100)', ...
      'BranchNames', parser.Results.BranchNames, ...
      'Degeneracy', parser.Results.Degeneracy);

  si.a0 = 5.431e-10;
  si.crystal_orientation = '100';
end
