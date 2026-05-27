"""
Run this cell first every time a SageMaker instance starts.
Assumes clearml.conf is already saved on the instance (~/.clearml.conf).
"""

import subprocess
import sys


def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        raise RuntimeError(f"Failed: {cmd}\n{result.stderr}")


run(f"{sys.executable} -m pip install -q clearml boto3 opencv-python-headless scipy scikit-learn torch torchvision")

from clearml import Task

task = Task.init(
    project_name="VitalLens",
    task_name="sagemaker-connectivity-check",
    task_type=Task.TaskTypes.testing,
    reuse_last_task_id=True,
)
print(f"ClearML connected — Task: {task.id}")
print(f"View at: {task.get_output_log_web_page()}")
task.close()

print("Setup complete.")
