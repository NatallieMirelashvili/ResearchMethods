from utils.job_manager import *

USER_NAME = "avivyuv"

# path to the main script to be executed
main_path = f"/home/{USER_NAME}/bigearthnet_v2/ResearchMethods/code/load_data.py"
num_jobs = 1

manager = JobManager(main_path, num_jobs)
manager.create_or_clear_log_file()
manager.create_jobs() 
