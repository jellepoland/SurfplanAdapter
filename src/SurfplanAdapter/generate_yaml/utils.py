import yaml
import re


def represent_list(self, data):
    # Force inline (flow) style for data arrays
    # Check if this is a data row (list with mixed types including dicts)
    if data and len(data) <= 10:  # Typical data row length
        # Special handling for rows that contain dictionaries
        if any(isinstance(item, dict) for item in data):
            # For wing_airfoils data rows with info_dict
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )
        # For simple numeric/string data rows
        elif all(isinstance(item, (int, float, str)) for item in data):
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )
    return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


def represent_none(self, data):
    return self.represent_scalar("tag:yaml.org,2002:null", "")


def save_to_yaml(yaml_data, yaml_file_path):
    yaml.add_representer(type(None), represent_none)
    yaml.add_representer(list, represent_list)

    with open(yaml_file_path, "w") as f:
        yaml.dump(
            yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # Post-process to clean up comments (remove quotes and None values)
    with open(yaml_file_path, "r") as f:
        content = f.read()

    # Clean up comment formatting
    content = content.replace("'#", "#")
    content = content.replace("': null", "")
    content = content.replace("': ''", "")

    # Convert empty line keys to actual blank lines - handle all patterns

    # Remove lines that are just empty keys with various patterns
    content = re.sub(r"^\? ''\n:\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"^\? ' '\n:\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"^\? '  '\n:\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"^\? '   '\n:\n", "\n", content, flags=re.MULTILINE)

    # Also handle simple key patterns
    content = re.sub(r"^'': *$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^' ': *$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^'  ': *$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^'   ': *$", "", content, flags=re.MULTILINE)

    with open(yaml_file_path, "w") as f:
        f.write(content)

    print(f'Generated YAML file and saved at "{yaml_file_path}"')
