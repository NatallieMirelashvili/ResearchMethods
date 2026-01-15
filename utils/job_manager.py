import os
import shutil
import sys
import re
import subprocess
import math
from typing import List, Dict, Set, Optional


# --------------------------------------------------------------------
# Helpers to query SLURM
# --------------------------------------------------------------------
def get_partition_nodes(partition: str) -> Set[str]:
    """
    Return the set of hostnames that belong to the given SLURM partition.
    """
    cmd = f'sinfo -h --Node --partition={partition} --Format="NodeHost"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running sinfo for partition {partition}: {result.stderr}")

    nodes: Set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        hostname = line.split()[0]
        nodes.add(hostname)
    return nodes


def get_top_n_nodes_with_max_cpus_and_mem(
    n: int = 10,
    cpu_partition: str = "cpu",
    gpu_partition: str = "gpu",
    reserve_cpus_per_node: int = 4,
    mem_fraction: float = 0.9,
    min_mem_gb: int = 16,
) -> List[Dict[str, int]]:
    """
    Returns top N *CPU-only* nodes (no GPUs) sorted by idle CPUs, filtered by min memory.

    Each returned dict has:
        {
            'hostname': <str>,
            'cpus': <usable idle CPUs>,
            'mem': <usable free memory in GB>
        }

    Notes:
      - FreeMem from sinfo is typically reported in MB.
      - We only keep nodes with usable free memory >= min_mem_gb (after mem_fraction).
    """
    cpu_nodes = get_partition_nodes(cpu_partition)
    gpu_nodes = get_partition_nodes(gpu_partition)
    cpu_only_nodes = cpu_nodes - gpu_nodes

    if not cpu_only_nodes:
        raise RuntimeError("No CPU-only nodes found (cpu partition minus gpu partition is empty).")

    cmd = (
        f'sinfo -r --Node --partition={cpu_partition} '
        '--exact --Format="NodeHost,CPUsState,FreeMem"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running sinfo command: {result.stderr}")

    nodes: List[Dict[str, int]] = []
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

        if hostname not in cpu_only_nodes:
            continue

        cpus_idle = int(info["cpus_idle"])
        free_mem_mb = int(info["free_mem"])

        usable_cpus = max(cpus_idle - reserve_cpus_per_node, 0)
        if usable_cpus <= 0:
            continue

        # Convert MB -> GB and apply safety fraction
        mem_gb = math.floor((free_mem_mb / 1024.0) * mem_fraction)
        if mem_gb < min_mem_gb:
            continue

        nodes.append({
            "hostname": hostname,
            "cpus": usable_cpus,
            "mem": mem_gb,
        })

    nodes_sorted = sorted(nodes, key=lambda x: x["cpus"], reverse=True)
    return nodes_sorted[:n]


# --------------------------------------------------------------------
# Job Manager
# --------------------------------------------------------------------
class JobManager:
    def __init__(
        self,
        main_path: str,
        num_jobs: int,
        parent_path: str = os.getcwd(),
        env_path: str = sys.executable,
        log_data_path: str = os.path.join(os.getcwd(), "main_logs"),
        partition: str = "cpu",
        max_cpus_per_job: int = 64,
        total_cpu_limit: int = 2600 - 128,
        min_mem_gb_per_job: int = 16,
    ):
        """
        main_path: path to the Python script to run.
        num_jobs:  number of jobs to submit.
        partition: SLURM partition to use (default: 'cpu').
        max_cpus_per_job: upper bound on CPUs per job to avoid huge jobs.
        total_cpu_limit: global CPU limit across all jobs (cluster-wide cap).
        min_mem_gb_per_job: minimum memory request per job (SBATCH --mem), in GB.
        """
        self.main_path = main_path
        self.parent_path = parent_path
        self.env_path = env_path
        self.log_data_path = log_data_path
        self.partition = partition
        self.max_cpus_per_job = max_cpus_per_job
        self.min_mem_gb_per_job = int(min_mem_gb_per_job)

        # CPU-only nodes, filtered to those with >= min_mem_gb_per_job free (after fraction)
        self.free_resources = get_top_n_nodes_with_max_cpus_and_mem(
            n=num_jobs,
            cpu_partition=self.partition,
            gpu_partition="gpu",
            min_mem_gb=self.min_mem_gb_per_job,
        )

        if len(self.free_resources) < num_jobs:
            print(
                f"[JobManager] Warning: requested {num_jobs} jobs but only found "
                f"{len(self.free_resources)} eligible CPU-only nodes with >= {self.min_mem_gb_per_job}GB free mem.",
                flush=True,
            )

        self.num_jobs = num_jobs
        self.job_scripts: List[str] = []
        self.job_ids: List[str] = []

        self.limit_cpu = total_cpu_limit
        self.current_limit_cpu = total_cpu_limit

        os.makedirs(self.log_data_path, exist_ok=True)

    def create_job_script(self, job_index: int) -> Optional[str]:
        """
        Create a single job script file and return its path.
        Returns None if no CPUs left to assign or index is out of eligible nodes.
        """
        if job_index >= len(self.free_resources):
            return None

        node_info = self.free_resources[job_index]
        mem_available = int(node_info["mem"])
        cpus_from_node = int(node_info["cpus"])

        # Always request at least 16GB (or configured minimum)
        mem_job = max(self.min_mem_gb_per_job, mem_available)

        print(
            f"[JobManager] Job {job_index}: node={node_info['hostname']} "
            f"cpus_available={cpus_from_node} mem_available={mem_available}G "
            f"mem_request={mem_job}G",
            flush=True
        )

        # Cap CPUs per job and by remaining global limit
        cpus_job = min(cpus_from_node, self.max_cpus_per_job, self.current_limit_cpu)
        if cpus_job <= 0:
            return None

        self.current_limit_cpu -= cpus_job

        job_script_name = os.path.join(self.log_data_path, f"job_script_{job_index}.sh")

        with open(job_script_name, "w") as script_file:
            script_file.write("#!/bin/bash\n")
            script_file.write(f"#SBATCH --job-name=multi_job_{job_index}\n")
            script_file.write(f"#SBATCH --cpus-per-task={cpus_job}\n")
            script_file.write(f"#SBATCH --mem={mem_job}G\n")
            script_file.write(
                f"#SBATCH --output={os.path.join(self.log_data_path, f'%j_job_output_{job_index}.txt')}\n"
            )
            script_file.write(f"#SBATCH --partition={self.partition}\n")
            script_file.write(f"export PYTHONPATH={self.parent_path}:$PYTHONPATH\n")
            script_file.write(f"{self.env_path} {self.main_path}\n")

        return job_script_name

    def create_jobs(self):
        """
        Create scripts for up to num_jobs jobs and submit them.
        """
        for job_index in range(self.num_jobs):
            job_script_name = self.create_job_script(job_index)
            if job_script_name is not None:
                self.job_scripts.append(job_script_name)

        if not self.job_scripts:
            print("[JobManager] No job scripts created (no eligible resources).", flush=True)
            return

        self.submit_jobs()
        self.cleanup()

    def submit_jobs(self):
        """
        Submit all created job scripts with sbatch.
        """
        for script in self.job_scripts:
            result = subprocess.run(
                ["sbatch", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            if result.returncode != 0:
                print(f"sbatch error for {script}:", result.stderr, file=sys.stderr)
                continue

            out = result.stdout.strip().split()
            if not out:
                print(f"Unexpected sbatch output for {script}: {result.stdout}", file=sys.stderr)
                continue

            job_id = out[-1]
            self.job_ids.append(job_id)

        print("Submitted jobs with IDs:", self.job_ids, flush=True)

    def cleanup(self):
        """
        Remove the temporary job script files.
        """
        for script in self.job_scripts:
            try:
                os.remove(script)
            except OSError:
                pass

    @staticmethod
    def create_or_clear_log_file(relative_log_file="../main_logs/error.txt"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(script_dir, relative_log_file)
        error_folder = os.path.join(script_dir, "../main_logs/error_folder")

        os.makedirs(error_folder, exist_ok=True)

        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            existing_files = [
                f for f in os.listdir(error_folder)
                if f.startswith("error_") and f.endswith(".txt")
            ]
            if existing_files:
                latest_file_num = max(int(f.split('_')[1].split('.')[0]) for f in existing_files)
            else:
                latest_file_num = 0

            new_file_num = latest_file_num + 1
            new_file_name = f"error_{new_file_num}.txt"
            new_file_path = os.path.join(error_folder, new_file_name)
            shutil.move(log_file, new_file_path)

        with open(log_file, 'w') as file:
            pass


# --------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------
if __name__ == "__main__":
    main_path = ""  # set your script path here
    log_dir = "main_logs"
    parent_path = os.getcwd()
    env_path = sys.executable
    log_data_path = os.path.join(parent_path, log_dir)

    num_jobs = 10

    manager = JobManager(
        main_path=main_path,
        num_jobs=num_jobs,
        parent_path=parent_path,
        env_path=env_path,
        log_data_path=log_data_path,
        partition="cpu",
        max_cpus_per_job=64,
        total_cpu_limit=2600 - 128,
        min_mem_gb_per_job=16,  # requests at least 16GB per job
    )
    manager.create_jobs()
