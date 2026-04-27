import os
import sys

import runpod
from dotenv import load_dotenv


def terminate_pod():
    # Load .env if it exists (for local testing/debugging)
    load_dotenv()

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY not found in environment.")
        return False

    pod_id = os.getenv("RUNPOD_POD_ID") or os.getenv("POD_ID")
    if not pod_id:
        print("RUNPOD_POD_ID not found in environment. Cannot terminate this pod automatically.")
        return False

    print(f"Terminating RunPod instance: {pod_id}...")

    runpod.api_key = api_key

    # Terminate the pod by its ID
    runpod.terminate_pod(pod_id)


if __name__ == "__main__":
    if terminate_pod():
        sys.exit(0)
    else:
        sys.exit(1)
