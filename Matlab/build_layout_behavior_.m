function mesh = build_layout_behavior_(mesh, layout)
% build_layout_behavior_ Attach rule-driven face semantics to mesh.

  mesh.boundary = struct('by_face', struct());
  mesh.face_rules = local_group_rules_by_normal_(layout);

  faces = {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'};
  for i = 1:numel(faces)
      tag = faces{i};
      if isfield(layout, 'boundary_patches') && isfield(layout.boundary_patches, tag)
          mesh.boundary.by_face.(tag) = layout.boundary_patches.(tag);
      else
          mesh.boundary.by_face.(tag) = [];
      end
  end

  % GENERATE/CATCH are handled directly as face actions now.
end

function grouped = local_group_rules_by_normal_(layout)
  grouped = struct();
  grouped.by_normal = struct( ...
      'xp', repmat(local_empty_face_rule_(), 0, 1), ...
      'xn', repmat(local_empty_face_rule_(), 0, 1), ...
      'yp', repmat(local_empty_face_rule_(), 0, 1), ...
      'yn', repmat(local_empty_face_rule_(), 0, 1), ...
      'zp', repmat(local_empty_face_rule_(), 0, 1), ...
      'zn', repmat(local_empty_face_rule_(), 0, 1));
  grouped.all = repmat(local_empty_face_rule_(), 0, 1);

  if ~isfield(layout, 'rules') || isempty(layout.rules)
      return;
  end

  for i = 1:numel(layout.rules)
      rule = layout.rules(i);
      face_rule = local_empty_face_rule_();
      face_rule.normal = upper(rule.normal);
      face_rule.mode = upper(rule.mode);
      face_rule.axis = rule.axis;
      face_rule.coord = rule.coord;
      face_rule.bounds = rule.bounds;
      face_rule.bounds_input = rule.bounds_input;
      face_rule.location = rule.location;
      face_rule.face_tag = rule.face_tag;
      face_rule.raw = rule.raw;

      grouped.all(end + 1, 1) = face_rule; %#ok<AGROW>
      key = local_normal_key_(face_rule.normal);
      grouped.by_normal.(key)(end + 1, 1) = face_rule; %#ok<AGROW>
  end
end

function key = local_normal_key_(normal_name)
  switch upper(normal_name)
    case '+X'
      key = 'xp';
    case '-X'
      key = 'xn';
    case '+Y'
      key = 'yp';
    case '-Y'
      key = 'yn';
    case '+Z'
      key = 'zp';
    case '-Z'
      key = 'zn';
    otherwise
      error('build_layout_behavior_: unsupported normal %s', normal_name);
  end
end

function rule = local_empty_face_rule_()
  rule = struct('normal', '', ...
                'mode', '', ...
                'axis', '', ...
                'coord', NaN, ...
                'bounds', zeros(1, 6), ...
                'bounds_input', zeros(1, 6), ...
                'location', '', ...
                'face_tag', '', ...
                'raw', '');
end
