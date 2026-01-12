import os
import shutil
import sys
import re
import subprocess
from typing import List, Dict, Set

# --------------------------------------------------------------------
# 1. Helpers to query SLURM
# --------------------------------------------------------------------

def get_top_n_gpu_nodes(
    n: int = 10,
    partition: str = "main",
    reserve_cpus_per_node: int = 2,
    mem_fraction: float = 0.9,
) -> List[Dict[str, any]]:
    """
    Returns top N nodes in the GPU partition sorted by idle CPUs.
    """
    cmd = (
        f'sinfo -r --Node --partition={partition} '
        '--exact --Format="NodeHost,CPUsState,FreeMem"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running sinfo: {result.stderr}")

    nodes: List[Dict[str, any]] = []
    seen_nodes: Set[str] = set()

    pattern = re.compile(
        r'(?P<hostname>\S+)\s+'
        r'(?P<cpus_alloc>\d+)/(?P<cpus_idle>\d+)/(?P<cpus_other>\d+)/(?P<cpus_total>\d+)\s+'
        r'(?P<free_mem>\d+)'
    )

    for line in result.stdout.splitlines():
        m = pattern.match(line)
        if not m:
            continue

        info = m.groupdict()
        hostname = info["hostname"]
        if hostname in seen_nodes or hostname == "HOSTNAMES":
            continue
        seen_nodes.add(hostname)

        cpus_idle = int(info["cpus_idle"])
        free_mem_mb = int(info["free_mem"])

        usable_cpus = max(cpus_idle - reserve_cpus_per_node, 0)
        mem_gb = int((free_mem_mb / 1024.0) * mem_fraction)

        if usable_cpus > 0 and mem_gb > 0:
            nodes.append({
                "hostname": hostname,
                "cpus": usable_cpus,
                "mem": mem_gb,
            })

    return sorted(nodes, key=lambda x: x["cpus"], reverse=True)[:n]


# --------------------------------------------------------------------
# 2. Job Manager
# --------------------------------------------------------------------

class JobManager:
    def __init__(
        self,
        main_path: str,
        num_jobs: int,
        parent_path: str = os.getcwd(),
        env_path: str = sys.executable,
        log_data_path: str = os.path.join(os.getcwd(), "main_logs"),
        partition: str = "main",
        qos: str = "normal",
        gpu_type: str = "rtx_3090:1",
        max_cpus_per_job: int = 4,
        total_cpu_limit: int = 100,
    ):
        self.main_path = main_path
        self.parent_path = parent_path
        self.env_path = env_path
        self.log_data_path = log_data_path
        self.partition = partition
        self.qos = qos
        self.gpu_type = gpu_type
        self.max_cpus_per_job = max_cpus_per_job
        self.current_limit_cpu = total_cpu_limit

        # Get nodes specifically for the GPU partition
        self.free_resources = get_top_n_gpu_nodes(
            n=num_jobs,
            partition=self.partition
        )

        self.num_jobs = min(num_jobs, len(self.free_resources))
        self.job_scripts: List[str] = []
        self.job_ids: List[str] = []

        os.makedirs(self.log_data_path, exist_ok=True)

    def create_job_script(self, job_index: int) -> str:
        node_info = self.free_resources[job_index]
        mem = int(node_info["mem"])
        cpus_available = node_info["cpus"]

        # Calculate how many CPUs to actually request
        cpus_job = min(cpus_available, self.max_cpus_per_job, self.current_limit_cpu)

        if cpus_job <= 0:
            print(f"Skipping job {job_index}: No CPUs available in limit.")
            return None

        self.current_limit_cpu -= cpus_job
        job_script_name = os.path.join(self.log_data_path, f"job_script_{job_index}.sh")

        with open(job_script_name, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name=gpu_job_{job_index}\n")
            f.write(f"#SBATCH --partition={self.partition}\n")
            f.write(f"#SBATCH --qos={self.qos}\n")
            f.write(f"#SBATCH --gres=gpu:{self.gpu_type}\n")
            f.write(f"#SBATCH --cpus-per-task={cpus_job}\n")
            f.write(f"#SBATCH --mem={mem}G\n")
            f.write(f"#SBATCH --time=06:00:00\n")
            f.write(f"#SBATCH --output={os.path.join(self.log_data_path, f'slurm-%j_job_{job_index}.txt')}\n")
            
            f.write(f"export PYTHONPATH={self.parent_path}:$PYTHONPATH\n")
            f.write(f"{self.env_path} {self.main_path}\n")

        return job_script_name

    def create_jobs(self):
        for job_index in range(self.num_jobs):
            script_path = self.create_job_script(job_index)
            if script_path:
                self.job_scripts.append(script_path)

        self.submit_jobs()
        self.cleanup()

    def submit_jobs(self):
        for script in self.job_scripts:
            result = subprocess.run(
                ["sbatch", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                self.job_ids.append(job_id)
                print(f"Submitted job {job_id} for script {script}")
            else:
                print(f"Error submitting {script}: {result.stderr}")

    def cleanup(self):
        for script in self.job_scripts:
            if os.path.exists(script):
                os.remove(script)

# --------------------------------------------------------------------
# 3. Execution
# --------------------------------------------------------------------

if __name__ == "__main__":
    target_python_script = "my_experiment.py" 

    manager = JobManager(
        main_path=target_python_script,
        num_jobs=5,
        partition="main",
        qos="normal",
        gpu_type="rtx_3090:1",
        max_cpus_per_job=4,
        total_cpu_limit=100
    )
    manager.create_jobs()