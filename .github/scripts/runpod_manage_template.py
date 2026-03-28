#!/usr/bin/env python3

import runpod
import os
from runpod.api import graphql


def parse_extra_envs(extra_envs_str):
    """
    Parse extra environment variables from flexible format.
    Supports:
    - Space-separated: KEY1=VALUE1 KEY2=VALUE2
    - Comma-separated: KEY1=VALUE1,KEY2=VALUE2
    - Newline-separated:
      KEY1=VALUE1
      KEY2=VALUE2
    - Mixed formats

    Returns: List of dicts [{"key": "KEY1", "value": "VALUE1"}, ...]
    """
    if not extra_envs_str or not extra_envs_str.strip():
        return []

    env_vars = []

    # Replace commas and newlines with spaces for uniform parsing
    normalized = extra_envs_str.replace(',', ' ').replace('\n', ' ')

    # Split by spaces and filter empty strings
    pairs = [pair.strip() for pair in normalized.split() if pair.strip()]

    for pair in pairs:
        if '=' not in pair:
            print(f"⚠️  Skipping invalid env var (no '='): {pair}")
            continue

        key, value = pair.split('=', 1)  # Split only on first '='
        key = key.strip()
        value = value.strip()

        if not key:
            print(f"⚠️  Skipping env var with empty key: {pair}")
            continue

        env_vars.append({"key": key, "value": value})

    return env_vars


if __name__ == '__main__':
    runpod.api_key = os.environ['RUNPOD_API_KEY']

    TEMPLATE_NAME = os.environ['TEMPLATE_NAME']
    IMAGE_URI = os.environ['IMAGE_URI']
    REGISTRY_ID = os.environ['REGISTRY_ID']
    VOLUME_SIZE_GB = int(os.environ['VOLUME_SIZE_GB'])
    VOLUME_MOUNT_PATH = os.environ['VOLUME_MOUNT_PATH']
    OMP_NUM_THREADS = os.environ['OMP_NUM_THREADS']
    MKL_NUM_THREADS = os.environ['MKL_NUM_THREADS']
    EXTRA_ENVS_STR = os.environ['EXTRA_ENVS_STR']

    print(f"Managing template: {TEMPLATE_NAME}")

    # Get existing templates
    query = """
    query {
        myself {
            podTemplates {
                id
                name
            }
        }
    }
    """

    result = graphql.run_graphql_query(query)
    templates = result["data"]["myself"]["podTemplates"]

    found_template = None
    for tmpl in templates:
        if tmpl["name"] == TEMPLATE_NAME:
            found_template = tmpl
            break

    # Build mutation
    id_field = f'id: "{found_template["id"]}"' if found_template else ''

    # Base environment variables
    base_env_vars = [
        {"key": "PYTHONUNBUFFERED", "value": "1"},
        {"key": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
        {"key": "FLAGS_selected_gpus", "value": "0"},
        {"key": "FLAGS_use_nccl", "value": "False"},
        {"key": "OMP_NUM_THREADS", "value": str(OMP_NUM_THREADS)},
        {"key": "MKL_NUM_THREADS", "value": str(MKL_NUM_THREADS)},
        {"key": "USE_GPU", "value": "true"},
        {"key": "FRAME_FPS", "value": "10"},
        {"key": "IS_LOWER_CASE", "value": "true"},
        {"key": "MAX_SEGMENT_SECONDS", "value": "10"},
    ]

    # Parse extra environment variables
    extra_env_vars = parse_extra_envs(EXTRA_ENVS_STR)

    if extra_env_vars:
        print(f"Adding {len(extra_env_vars)} extra environment variables:")
        for env_var in extra_env_vars:
            print(f"   {env_var['key']}={env_var['value']}")

    # Merge environment variables (extra vars can override base vars)
    all_env_vars = base_env_vars.copy()

    # Check for overrides
    base_keys = {var["key"] for var in base_env_vars}
    for extra_var in extra_env_vars:
        if extra_var["key"] in base_keys:
            print(f"⚠️  Overriding base env var: {extra_var['key']}")
            # Remove base var
            all_env_vars = [v for v in all_env_vars if v["key"] != extra_var["key"]]
        all_env_vars.append(extra_var)

    # Format env vars for GraphQL
    env_vars_graphql = ", ".join([
        f'{{key: "{var["key"]}", value: "{var["value"]}"}}'
        for var in all_env_vars
    ])

    mutation = f"""
    mutation {{
        saveTemplate(input: {{
            {id_field}
            name: "{TEMPLATE_NAME}"
            imageName: "{IMAGE_URI}"
            dockerArgs: ""
            containerRegistryAuthId: "{REGISTRY_ID}"
            containerDiskInGb: 30
            volumeInGb: {VOLUME_SIZE_GB}
            volumeMountPath: "{VOLUME_MOUNT_PATH}"
            ports: "8000/tcp"
            env: [{env_vars_graphql}]
        }}) {{
            id
            name
        }}
    }}
    """

    result = graphql.run_graphql_query(mutation)
    template_id = result["data"]["saveTemplate"]["id"]

    if found_template:
        print(f"✅ Updated template: {template_id}")
    else:
        print(f"✅ Created template: {template_id}")

    # Save output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f"template_id={template_id}\n")

