from azureml.core.workspace import Workspace
import urllib
import tarfile
import os
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import PyTorchConfiguration
ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')
project_folder = 'src'
experiment_name = 'pytorch-distr'
experiment = Experiment(ws, name=experiment_name)
pytorch_env = Environment.get(ws, name='AzureML-PyTorch-1.6-GPU')

# create distributed config
distr_config = PyTorchConfiguration(node_count=2)

# define command
launch_cmd = ["pip install azureml-mlflow && python -m torch.distributed.launch --nproc_per_node 1 --nnodes 2 " \
    "--node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --use_env " \
    "train.py --data-dir cifar-10 --epochs 20"]

# create job config
src = ScriptRunConfig(source_directory=project_folder,
                    command=launch_cmd,
                    compute_target="gpucluster-1x",
                    environment=pytorch_env,
                    distributed_job_config=distr_config)
run=experiment.submit(src)
print(run)