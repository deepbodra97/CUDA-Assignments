



To request allocation on GPU partition for 1 hour

   srun -p gpu --nodes=1 --gpus=geforce:1 --time=01:00:00  --pty -u bash -i



Command to load necessary modules

   module load ufrc cmake/3.19.1 intel/2018.1.163 cuda/10.0.130 git



Use git to clone /blue/cis4936/share

   git clone /blue/cis4936/share

   Then you can get updates with 'git pull'.  (Don't try to push)



How to use cmake to create a makefile

  1. create a directory for your build artifacts:  mkdir build
  2. enter your new directory:  cd build
  3  create the makefile:  cmake <dir containing  CMakeLists.txt>
      If CMakeLists.txt is in the parent directory of build:  cmake ..
      Alternatively, running  'ccmake ..' will provide "gui"  that shows default configuration parameters and allows you to change them. 
      Before doing you final timings, you might want to change the build type from Debug to Release.
  4. compile:  make
  5. execute: 
       ./deviceQuery 
       ./transpose 



CUDA Documentation for version 10.0

  https://docs.nvidia.com/cuda/archive/10.0/


Possibly useful commands

nvidia-smi
  Shows devices installed on machine.  (cudaGetDeviceCount only counts devices that are available and not in use by others, not how many exist on machine)

cuda-gdb
  Debugger gdb extended to handle CUDA 


