function materials = resolve_case_materials_(cs)
% resolve_case_materials_ Load every material referenced by the parsed case layout.
%
% This is the central material registry used by geometry-to-material mapping.
% Even when the current solver only advances one spectral grid, all requested
% materials are still loaded here so the case data model stays multi-material ready.

  requested_names = local_case_material_names_(cs);
  if isempty(requested_names)
      requested_names = {'SILICON'};
  end

  entries = repmat(local_empty_material_entry_(), numel(requested_names), 1);
  for i = 1:numel(requested_names)
      raw_name = char(requested_names{i});
      key = local_material_key_(raw_name);
      entries(i).name = raw_name;
      entries(i).key = key;
      entries(i).mat = local_load_material_(key, raw_name);
  end

  primary_name = local_primary_material_(cs, requested_names);
  primary_key = local_material_key_(primary_name);
  primary_index = find(strcmp({entries.key}, primary_key), 1, 'first');
  if isempty(primary_index)
      primary_index = 1;
      primary_name = entries(1).name;
      primary_key = entries(1).key;
  end

  materials = struct();
  materials.names = {entries.name};
  materials.keys = {entries.key};
  materials.list = entries;
  materials.by_key = local_entries_to_struct_(entries);
  materials.primary_name = primary_name;
  materials.primary_key = primary_key;
  materials.primary_index = primary_index;
  [materials.region_material_name, materials.region_material_index] = ...
      local_region_material_index_(cs, entries);
end

function names = local_case_material_names_(cs)
  raw_names = {};
  if isfield(cs, 'regions') && ~isempty(cs.regions)
      raw_names = {cs.regions.material};
  elseif isfield(cs, 'materials') && ~isempty(cs.materials)
      raw_names = cs.materials;
  end

  if isempty(raw_names)
      names = {};
      return;
  end

  names = {};
  seen = {};
  for i = 1:numel(raw_names)
      name_i = char(raw_names{i});
      key_i = local_material_key_(name_i);
      if ~any(strcmp(seen, key_i))
          names{end + 1} = name_i; %#ok<AGROW>
          seen{end + 1} = key_i; %#ok<AGROW>
      end
  end
end

function primary_name = local_primary_material_(cs, fallback_names)
  primary_name = '';
  if isfield(cs, 'regions') && ~isempty(cs.regions) && isfield(cs.regions(1), 'material')
      primary_name = cs.regions(1).material;
  elseif isfield(cs, 'materials') && ~isempty(cs.materials)
      primary_name = cs.materials{1};
  elseif ~isempty(fallback_names)
      primary_name = fallback_names{1};
  end

  if isempty(primary_name)
      primary_name = 'SILICON';
  end
end

function key = local_material_key_(name)
  key = upper(strtrim(char(string(name))));
  switch key
    case {'SI', 'SILICON', 'SI_100', 'SILICON_100'}
      key = 'SILICON';
    case 'IGZO'
      key = 'IGZO';
  end
end

function mat = local_load_material_(key, raw_name)
  switch key
    case 'SILICON'
      mat = mat_silicon_100();
    case 'IGZO'
      mat = mat_igzo();
    otherwise
      warning('resolve_case_materials_: unsupported material "%s". Falling back to Silicon.', raw_name);
      mat = mat_silicon_100();
  end
  mat.case_material = key;
  mat.case_material_label = raw_name;
end

function s = local_entries_to_struct_(entries)
  s = struct();
  for i = 1:numel(entries)
      field_name = matlab.lang.makeValidName(entries(i).key);
      s.(field_name) = entries(i).mat;
  end
end

function [region_names, region_index] = local_region_material_index_(cs, entries)
  if ~isfield(cs, 'regions') || isempty(cs.regions)
      region_names = {};
      region_index = zeros(0, 1, 'int32');
      return;
  end

  region_names = {cs.regions.material};
  region_index = zeros(numel(region_names), 1, 'int32');
  for ir = 1:numel(region_names)
      key = local_material_key_(region_names{ir});
      idx = find(strcmp({entries.key}, key), 1, 'first');
      if ~isempty(idx)
          region_index(ir) = int32(idx);
      end
  end
end

function entry = local_empty_material_entry_()
  entry = struct('name', '', 'key', '', 'mat', struct());
end
