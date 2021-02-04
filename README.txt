


To request allocation on GPU partition for 1 hour

srun -p gpu --nodes=1 --gpus=geforce:1 --time=01:00:00  --pty -u bash -i



Command to load necessary modules

module load ufrc cmake/3.19.1 intel/2018.1.163 cuda/10.0.130 git



CUDA Documentation for version 10.0

https://docs.nvidia.com/cuda/archive/10.0/

Possibly useful commands

nvidia-smi
  Shows devices installed on machine.  (cudaGetDeviceCount only counts devices that are available and not in use by others, not how many exist on machine)

cuda-gdb
  Debugger gdb extended to handle CUDA 


