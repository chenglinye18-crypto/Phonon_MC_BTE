function v = getfield_or(s, name, default_v)
%GETFIELD_OR  安全读取结构体字段；不存在或为空则给默认值
%   v = getfield_or(s, 'field', default)
  if isstruct(s) && isfield(s, name)
      val = s.(name);
      if ~isempty(val)
          v = val;
          return;
      end
  end
  v = default_v;
end
