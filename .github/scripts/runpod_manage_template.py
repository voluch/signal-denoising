#!/usr/bin/env python3

import re
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
    - JSON object values: KEY1={"a": 1, "b": 2} KEY2=simple
    - Quoted values: KEY1="some value" KEY2='other value'
    - Mixed formats

    Returns: List of dicts [{"key": "KEY1", "value": "VALUE1"}, ...]
    """
    if not extra_envs_str or not extra_envs_str.strip():
        return []

    env_vars = []

    # Match KEY=VALUE where VALUE is one of:
    #   {…}        JSON object (may contain spaces and commas)
    #   "…"        double-quoted string (supports \" escapes inside)
    #   '…'        single-quoted string (supports \' escapes inside)
    #   [^\s,]+    simple token (no whitespace or comma)
    pattern = re.compile(
        r'([A-Za-z_][A-Za-z0-9_]*)='
        r'(\{[^}]*\}|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^\s,]+)'
    )

    for match in pattern.finditer(extra_envs_str):
        key = match.group(1)
        value = match.group(2)

        # Strip wrapping quotes and unescape internal escape sequences
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1].replace('\\"', '"')
        elif len(value) >= 2 and value[0] == "'" and value[-1] == "'":
            value = value[1:-1].replace("\\'", "'")

        if not key:
            print(f"⚠️  Skipping env var with empty key: {match.group(0)}")
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
    DATASET_PATH = os.environ['DATASET_PATH']
    WANDB_PROJECT = os.environ['WANDB_PROJECT']
    EXTRA_DOCKER_ARGS = os.environ['EXTRA_DOCKER_ARGS']
    MODEL = os.environ.get('MODEL', 'all')
    RUN_ID = os.environ.get('RUN_ID', '')
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY', '')
    RUNPOD_API_KEY_SECRET = os.environ.get('RUNPOD_API_KEY', '')
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID', '')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
    S3_BUCKET = os.environ.get('S3_BUCKET', '')
    EXTRA_ENVS_STR = os.environ['EXTRA_ENVS_STR']
    RUNPOD_POD_ID = os.environ.get('RUNPOD_POD_ID', '')

    print(f"Managing template: {TEMPLATE_NAME}")

    # Build dockerArgs: --dataset <DATASET_PATH> --wandb-project <WANDB_PROJECT> --models <MODEL> <EXTRA_DOCKER_ARGS>
    docker_args = f"--dataset {DATASET_PATH}"
    if WANDB_PROJECT:
        docker_args += f" --wandb-project {WANDB_PROJECT}"
    if MODEL:
        docker_args += f" --models {MODEL}"
    if RUN_ID:
        docker_args += f" --run-id {RUN_ID}"
    if EXTRA_DOCKER_ARGS:
        docker_args += f" {EXTRA_DOCKER_ARGS}"

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

    # Base environment variables for the training script
    base_env_vars = [
        {"key": "PYTHONUNBUFFERED", "value": "1"},
        {"key": "CLOUD_TRAINING", "value": "True"},
    ]
    if WANDB_API_KEY:
        base_env_vars.append({"key": "WANDB_API_KEY", "value": WANDB_API_KEY})
    if RUNPOD_API_KEY_SECRET:
        base_env_vars.append({"key": "RUNPOD_API_KEY", "value": RUNPOD_API_KEY_SECRET})
    if AWS_ACCESS_KEY_ID:
        base_env_vars.append({"key": "AWS_ACCESS_KEY_ID", "value": AWS_ACCESS_KEY_ID})
    if AWS_SECRET_ACCESS_KEY:
        base_env_vars.append({"key": "AWS_SECRET_ACCESS_KEY", "value": AWS_SECRET_ACCESS_KEY})
    if S3_BUCKET:
        base_env_vars.append({"key": "S3_BUCKET", "value": S3_BUCKET})
    if RUNPOD_POD_ID:
        base_env_vars.append({"key": "RUNPOD_POD_ID", "value": RUNPOD_POD_ID})

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

    # Format env vars for GraphQL — escape backslashes then double quotes in values
    def _gql_escape(s):
        return s.replace('\\', '\\\\').replace('"', '\\"')

    env_vars_graphql = ", ".join([
        f'{{key: "{var["key"]}", value: "{_gql_escape(var["value"])}"}}'
        for var in all_env_vars
    ])

    mutation = f"""
    mutation {{
        saveTemplate(input: {{
            {id_field}
            name: "{TEMPLATE_NAME}"
            imageName: "{IMAGE_URI}"
            dockerArgs: "{docker_args}"
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

