#!/usr/bin/env python3

import os
import sys
import time
from typing import Optional, List, Dict, Any

import runpod
from runpod.api import graphql

PRIVATE_TCP_PORT = 8000

runpod.api_key = os.environ['RUNPOD_API_KEY']

POD_NAME = os.environ["POD_NAME"]
TEMPLATE_NAME = os.environ["TEMPLATE_NAME"]
TEMPLATE_ID = os.environ["TEMPLATE_ID"]
IMAGE_URI = os.environ["IMAGE_URI"]
GPU_TYPE = os.environ["GPU_TYPE"]
DATA_CENTER_ID = os.environ.get("DATA_CENTER_ID", "AUTO")
USE_SPOT = os.environ.get("USE_SPOT", "false").lower() == "true"
USE_NETWORK_VOLUME = os.environ.get("USE_NETWORK_VOLUME", "false").lower() == "true"
REPLACE = os.environ.get("REPLACE", "false").lower() == "true"
VOLUME_SIZE_GB = int(os.environ.get("VOLUME_SIZE_GB", "30"))
VOLUME_MOUNT_PATH = os.environ.get("VOLUME_MOUNT_PATH", "/app/data")
NETWORK_VOLUME_ID = os.environ.get("NETWORK_VOLUME_ID")

DEPLOY_SCOPE = os.environ.get("DEPLOY_SCOPE", "EU+US").upper()

# Build ordered list of GPU types to try: primary first, then fallbacks.
_gpu_fallback_raw = os.environ.get("GPU_FALLBACK", "").strip()
GPU_TYPES: List[str] = [GPU_TYPE]
if _gpu_fallback_raw:
    GPU_TYPES += [g.strip() for g in _gpu_fallback_raw.replace("\n", ",").split(",") if g.strip()]

# ----------------------------
# Region candidates
# IDs from RunPod docs / blog posts (runpod.io/blog/runpod-global-networking-expansion,
# runpod.io/blog/runpod-apac-launch-fukushima).
# ----------------------------
EU_DCS = [
    "EU-NL-1",
    "EU-RO-1",
    "EU-CZ-1",
    "EU-FR-1",
    "EU-SE-1",
    "EUR-IS-1",
    "EUR-IS-2",
    "EUR-IS-3",
    "EUR-NO-1",
]

US_DCS = [
    "US-IL-1",
    "US-TX-3",
    "US-TX-4",
    "US-KS-2",
    "US-GA-2",
    "US-WA-1",
    "US-CA-2",
    "US-DE-1",
    "US-MO-2",
    "US-NC-1",
]

CA_DCS = [
    "CA-MTL-4",
    "CA-MTL-3",
]

APAC_DCS = [
    "AP-JP-1",   # Japan (Fukushima) — first APAC datacenter
    "OC-AU-1",   # Australia
]


def write_github_output(**kwargs: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def get_candidate_datacenters(
        preferred_dc: str,
        deploy_scope: str,
) -> List[str]:
    if preferred_dc and preferred_dc != "AUTO":
        pool = EU_DCS + US_DCS + CA_DCS + APAC_DCS
        if preferred_dc in pool:
            return [preferred_dc] + [dc for dc in pool if dc != preferred_dc]
        return [preferred_dc]

    scope_map = {
        "EU":          EU_DCS,
        "US":          US_DCS,
        "CA":          CA_DCS,
        "APAC":        APAC_DCS,
        "EU+US":       EU_DCS + US_DCS,
        "EU+APAC":     EU_DCS + APAC_DCS,
        "US+APAC":     US_DCS + APAC_DCS,
        "EU+US+APAC":  EU_DCS + US_DCS + APAC_DCS,
        "ALL":         EU_DCS + US_DCS + CA_DCS + APAC_DCS,
    }
    return scope_map.get(deploy_scope, EU_DCS + US_DCS)[:]


def get_network_volume_datacenter(network_volume_id: str) -> str:
    query = f"""
    query {{
      myself {{
        networkVolumes {{
          id
          name
          dataCenterId
        }}
      }}
    }}
    """
    result = graphql.run_graphql_query(query)
    if result.get("errors"):
        raise RuntimeError(result["errors"][0]["message"])

    volumes = result["data"]["myself"]["networkVolumes"]
    for volume in volumes:
        if volume["id"] == network_volume_id:
            return volume["dataCenterId"]

    raise RuntimeError(f"Network volume not found: {network_volume_id}")


def get_pod_url(pod_obj: Dict[str, Any]) -> Optional[str]:
    runtime = pod_obj.get("runtime") or {}
    ports = runtime.get("ports") or []

    if not ports:
        print("Warning: Pod has no runtime ports yet")
        return None

    for port_info in ports:
        if port_info.get("privatePort") != PRIVATE_TCP_PORT:
            continue

        port_type = (port_info.get("type") or "").lower()

        if port_type == "http":
            pod_url = f"https://{pod_obj.get('id')}-{PRIVATE_TCP_PORT}.proxy.runpod.net"
            print(f"Type: HTTP Proxy")
            print(f"URL: {pod_url}")
            return pod_url

        if port_type == "tcp":
            ip = port_info.get("ip")
            public_port = port_info.get("publicPort")
            is_public = port_info.get("isIpPublic", False)

            if not ip or not public_port:
                print("Warning: TCP port info incomplete")
                return None

            pod_url = f"http://{ip}:{public_port}"
            print("Type: TCP Direct")
            print(f"IP: {ip} ({'Public' if is_public else 'Private'})")
            print(f"Port: {public_port} (mapped from {PRIVATE_TCP_PORT})")
            print(f"URL: {pod_url}")
            return pod_url

        print(f"Unknown port type: {port_type}")
        return None

    print(f"Warning: Port {PRIVATE_TCP_PORT} not found")
    print(f"Available ports: {[p.get('privatePort') for p in ports]}")
    return None


def wait_for_pod_runtime(pod_id: str, timeout_sec: int = 300, poll_sec: int = 5) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_pod = None

    while time.time() < deadline:
        try:
            pod = runpod.get_pod(pod_id)
            last_pod = pod
            runtime = pod.get("runtime") or {}
            ports = runtime.get("ports") or []
            if ports:
                return pod
        except Exception:
            pass

        time.sleep(poll_sec)

    return last_pod or {}


def resolve_pod_name() -> str:
    """
    Returns the pod name to deploy with:
    - No existing pod with POD_NAME → use POD_NAME as-is.
    - REPLACE=true and pod exists → terminate it, reuse POD_NAME.
    - REPLACE=false and pod exists → find next free versioned name
      (POD_NAME-2, POD_NAME-3, …) and deploy in parallel.
    """
    pods = runpod.get_pods()
    existing_names = {pod.get("name") for pod in pods}

    base_pod = next((p for p in pods if p.get("name") == POD_NAME), None)

    if base_pod is None:
        return POD_NAME

    print("Found existing pod:")
    print(f"  ID: {base_pod['id']}")
    print(f"  Status: {base_pod.get('desiredStatus', 'unknown')}")

    if REPLACE:
        print(f"Terminating existing pod: {base_pod['id']}")
        runpod.terminate_pod(base_pod["id"])
        time.sleep(20)
        return POD_NAME

    # Find next available versioned name
    n = 2
    while True:
        candidate = f"{POD_NAME}-{n}"
        if candidate not in existing_names:
            print(f"Pod '{POD_NAME}' already exists — deploying parallel pod as '{candidate}'")
            return candidate
        n += 1


def build_mutation_for_dc(dc: str, pod_name: str, gpu_type: str) -> str:
    mutation_name = "podRentInterruptable" if USE_SPOT else "podFindAndDeployOnDemand"
    bid_line = "bidPerGpu: 0.0" if USE_SPOT else ""

    if USE_NETWORK_VOLUME:
        if not NETWORK_VOLUME_ID:
            raise RuntimeError("USE_NETWORK_VOLUME=true but NETWORK_VOLUME_ID is missing")

        input_body = f"""
            {bid_line}
            cloudType: SECURE
            gpuCount: 1
            gpuTypeId: "{gpu_type}"
            name: "{pod_name}"
            templateId: "{TEMPLATE_ID}"
            dataCenterId: "{dc}"
            networkVolumeId: "{NETWORK_VOLUME_ID}"
        """
    else:
        input_body = f"""
            {bid_line}
            cloudType: SECURE
            gpuCount: 1
            gpuTypeId: "{gpu_type}"
            name: "{pod_name}"
            templateId: "{TEMPLATE_ID}"
            dataCenterId: "{dc}"
            volumeInGb: {VOLUME_SIZE_GB}
            volumeMountPath: "{VOLUME_MOUNT_PATH}"
        """

    return f"""
    mutation {{
      {mutation_name}(input: {{
        {input_body}
      }}) {{
        id
        desiredStatus
      }}
    }}
    """


def deploy_with_fallback(dcs_to_try: List[str], pod_name: str) -> Dict[str, Any]:
    """Try each GPU type across all candidate DCs before moving to the next GPU type."""
    mutation_name = "podRentInterruptable" if USE_SPOT else "podFindAndDeployOnDemand"
    last_error = None

    retryable_markers = [
        "not available",
        "no longer any instances",
        "insufficient",
        "capacity",
        "unavailable",
    ]

    for gpu_type in GPU_TYPES:
        print(f"\n── GPU: {gpu_type} ──")
        for dc in dcs_to_try:
            print(f"  Trying datacenter: {dc}")
            mutation = build_mutation_for_dc(dc, pod_name, gpu_type)

            try:
                result = graphql.run_graphql_query(mutation)

                if result.get("errors"):
                    raise RuntimeError(result["errors"][0]["message"])

                pod_data = result["data"][mutation_name]
                if not pod_data or not pod_data.get("id"):
                    raise RuntimeError(f"Create pod returned empty data for {dc}")

                print(f"  Deployment initiated — GPU: {gpu_type}, DC: {dc}")
                return {
                    "pod_id": pod_data["id"],
                    "final_dc": dc,
                    "final_gpu": gpu_type,
                    "desired_status": pod_data.get("desiredStatus", "unknown"),
                }

            except Exception as e:
                last_error = e
                msg = str(e).lower()
                print(f"  Failed ({dc}): {e}")

                if "network volume" in msg:
                    raise RuntimeError(
                        f"Network volume issue in {dc}: {e}. "
                        "Network volumes are datacenter-specific."
                    ) from e

                if any(marker in msg for marker in retryable_markers):
                    continue

                continue

    raise RuntimeError(f"All deployment attempts failed. Last error: {last_error}")


def manage_pod() -> None:
    print("=" * 60)
    print("DEPLOYMENT CONFIGURATION")
    print("=" * 60)
    print(f"Pod: {POD_NAME}")
    print(f"Template: {TEMPLATE_NAME}")
    print(f"Requested DC: {DATA_CENTER_ID}")
    print(f"Scope: {DEPLOY_SCOPE}")
    print(f"Image: {IMAGE_URI}")
    print(f"GPUs to try: {', '.join(GPU_TYPES)}")
    print(f"Type: {'Spot' if USE_SPOT else 'On-Demand'}")
    print(f"Replace: {REPLACE}")
    print(f"Use network volume: {USE_NETWORK_VOLUME}")
    print("=" * 60)

    actual_pod_name = resolve_pod_name()

    if USE_NETWORK_VOLUME:
        volume_dc = get_network_volume_datacenter(NETWORK_VOLUME_ID)
        dcs_to_try = [volume_dc]
        print(f"Network volume {NETWORK_VOLUME_ID} is pinned to {volume_dc}")
    else:
        dcs_to_try = get_candidate_datacenters(DATA_CENTER_ID, DEPLOY_SCOPE)

    print(f"Candidate datacenters: {', '.join(dcs_to_try)}")

    deployment = deploy_with_fallback(dcs_to_try, actual_pod_name)
    pod_id = deployment["pod_id"]
    final_dc = deployment["final_dc"]
    final_gpu = deployment["final_gpu"]

    print("Waiting for pod runtime...")
    full_pod = wait_for_pod_runtime(pod_id)

    pod_url = get_pod_url(full_pod) if full_pod else None

    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"Pod Name:      {actual_pod_name}")
    print(f"Pod ID:        {pod_id}")
    print(f"Pod URL:       {pod_url}")
    print(f"Template ID:   {TEMPLATE_ID}")
    print(f"GPU Type:      {final_gpu}")
    print(f"Instance Type: {'Spot' if USE_SPOT else 'On-Demand'}")
    print(f"Datacenter:    {final_dc}")
    print(f"Image:         {IMAGE_URI}")
    if USE_NETWORK_VOLUME:
        print(f"Network Volume:{NETWORK_VOLUME_ID}")
    print("=" * 60)

    write_github_output(
        pod_id=pod_id,
        pod_url=pod_url or "",
        action="deployed",
        template_id=TEMPLATE_ID,
        datacenter=final_dc,
        final_gpu=final_gpu,
        network_volume_id=NETWORK_VOLUME_ID or "",
    )

    if USE_SPOT:
        print("\nSPOT WARNING:")
        print("- Pod may be interrupted")
        print("- Save work frequently")


if __name__ == "__main__":
    manage_pod()
