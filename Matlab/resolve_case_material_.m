function mat = resolve_case_material_(cs)
% resolve_case_material_ Resolve the primary material while keeping the full library.
%
% The solver still builds a single global spectral grid today, but the returned
% material struct carries the complete material library so cell-wise dispatch can
% be added later without changing the case-loading interface.

  materials = resolve_case_materials_(cs);
  mat = materials.list(materials.primary_index).mat;
  mat.case_material = materials.primary_key;
  mat.case_material_label = materials.primary_name;
  mat.material_library = materials;
end
