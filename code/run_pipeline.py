import os
import subprocess
import sys

scripts_to_run = [
    "load_data.py",
    "preprocess.py",
    "train.py",
    "test.py"
]

def run_pipeline():
    cwd = os.path.dirname(os.path.abspath(__file__))

    print(f"🚀 Starting pipeline execution from: {cwd}")

    for script_name in scripts_to_run:
        script_path = os.path.join(cwd, script_name)
        print(f"--------------------------------------------------")
        print(f"▶️  Running script: {script_name}")
        print(f"--------------------------------------------------")
        
        try:
            subprocess.run([sys.executable, script_path], check=True, cwd=cwd)
            print(f"✅ Finished {script_name} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {script_name}. Pipeline stopped.")
            sys.exit(1) 

if __name__ == "__main__":
    run_pipeline()