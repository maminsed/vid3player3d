"""Config lookup and flat artifact names, without simulator dependencies."""
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'cfg'


def resolve_config(value, root=CONFIG_ROOT):
    """Accept an existing YAML path or a unique name relative to a cfg category.

    For example, controller/strategy/stage.yaml can be selected using
    strategy/stage, controller/strategy/stage, or its unique basename stage.
    """
    root = Path(root).resolve()
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_absolute():
        raise FileNotFoundError(f'Config does not exist: {path}')
    project_path = root.parent.parent / path
    if project_path.is_file():
        return project_path.resolve()
    name = path.with_suffix('') if path.suffix == '.yaml' else path
    matches = []
    for candidate in root.rglob('*.yaml'):
        relative = candidate.relative_to(root).with_suffix('')
        if (relative == name or Path(*relative.parts[1:]) == name
                or (len(name.parts) == 1 and candidate.stem == str(name))):
            matches.append(candidate.resolve())
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f'No config matches {value!r} under {root}')
    if len(matches) > 1:
        choices = ', '.join(str(p.relative_to(root)) for p in matches)
        raise ValueError(f'Ambiguous config {value!r}; use a qualified name or YAML path: {choices}')
    return matches[0]


def config_output_name(path, root=CONFIG_ROOT):
    """Flatten category-relative folders with __; external YAMLs use their stem."""
    path = Path(path).resolve()
    try:
        relative = path.relative_to(Path(root).resolve()).with_suffix('')
    except ValueError:
        return path.stem
    # controller and imitation configs have historically omitted the category.
    parts = relative.parts[1:] if len(relative.parts) > 1 else relative.parts
    return '__'.join(parts)
