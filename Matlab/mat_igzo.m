function mat = mat_igzo(varargin)
% mat_igzo Load IGZO phonon dispersion from text data.
%
% Default source file:
%   input/phonon_dispersion_IGZO.txt

  parser = inputParser;
  parser.FunctionName = mfilename;
  addParameter(parser, 'FilePath', fullfile('input', 'phonon_dispersion_IGZO.txt'), ...
      @(x) ischar(x) || isstring(x));
  addParameter(parser, 'BranchNames', {}, @(x) iscell(x) || isstring(x));
  addParameter(parser, 'Degeneracy', [], @isnumeric);
  parse(parser, varargin{:});

  mat = mat_from_phonon_dispersion_file_( ...
      'FilePath', parser.Results.FilePath, ...
      'MaterialName', 'IGZO', ...
      'BranchNames', parser.Results.BranchNames, ...
      'Degeneracy', parser.Results.Degeneracy);
end
