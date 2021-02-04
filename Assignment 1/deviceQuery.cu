#include <iostream>
#include "cudaCheck.cuh"



int main (int argc, char *argv[])
{

  int numDevices;
  cudaCheck( cudaGetDeviceCount(&numDevices));
  std::cout << "Number of devices: " << numDevices << std::endl;

 

  for (int devId = 0; devId < numDevices; devId++){
     std::cout << "-------------------------------------\n";
     cudaDeviceProp prop;
     cudaCheck( cudaGetDeviceProperties(&prop, devId));
     std::cout << "Device id: " << devId << std::endl;
     std::cout << "Device name: " << prop.name << std::endl;

/*@  Calculate and print the theoretical peak bandwidth of the
     CUDA device.  You will need to obtain additional information
     from prop.  See the documentation 
*/

  }
  std::cout << "-------------------------------------\n";

}

 
