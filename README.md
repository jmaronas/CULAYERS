# CULAYERS
This is culayers, a C++ wrapper over CUDA to easy deploy applications running on CUDA without caring about GPU managment.

Culayers is used by [layers](https://github.com/RParedesPalacios/Layers), an academic neural network toolkit which complements more sophisticated libraries such as PyTorch or TensorFlow, for those who want to learn or deploy neural networks easily.

Culayers is easy to use, just create the library and link your .cpp files with it. You do not need to be root of your system to use it. I now go over the different features and ways to use it, with some examples.

Culayers is ready to used with ubuntu16 and cuda10. If you want to use it in mac try and compile and generate and instruction file and make a pull request, or just send me and email and I will add it.

Culayers is my personal project. I used this project to learn CUDA programming and to help my advisor Roberto Paredes porting his academic software to have GPU computation. This means that you can probably find much better libraries such as THC from Facebook. The good point of this library is that for learn CUDA and to deploy simple application is easier to understand. Do not try to train a deep net with this, use directly PyTorch as example. 

## Compilation and Set Up

Compile culayers is really easy, just clone the repo and execute ```make```. You will get your library shared object ```libculayers.so```. Before that set PATH enviroment variable with the directory to the binaries of cuda, tipically /usr/local/cuda-8.0/bin so you have access to nvcc compiler.

The easiest way to use libculayers is to configure your enviroment correctly. To install libculayers in the whole system you can either add this file to ```/usr/lib64/``` or ```/lib64/```

You can set this enviroment as:

```export LD_LIBRARY_PATH:path_to_cuda_lib64:path_to_libculayer```

where path_to_cuda_lib64 is the absolute path to cuda lib64, tipically /usr/local/cuda-8.0/lib64/ and path_to_libculayer which by default is the absolute path to the GIT repo. 

Now assume your cpp file is example.cpp you have to do the next steps:

Headers files to include:
```
#include "gpu_env.h"
#include "utils.h"
```
And then to compile you have two options, compile and link with g++ or with nvcc:

With g++ you have to do
``` 
g++ -I /usr/local/cuda-8.0/include/ -L /usr/local/cuda-8.0/lib64/  example.cpp -o exe -Lpath_to_culayers -lculayers
```
and with nvcc just:
```
nvcc example.cpp -o exe -Lpath_to_culayers -lculayers
```
Then you can run your exe file and everything will run on GPU. Check file [library](https://github.com/jmaronas/CULAYERS/blob/master/library.md) for a description on the different functions and arguments. Chech [example](https://github.com/jmaronas/CULAYERS/tree/master/examples) for an example.

## Todo

compile in other ubuntus

cudnn and convolutions

pinned memory and n-dimensional arrays

some linear algebra operators

implement efficients algorithms for reduction operators

implement all the memory managment using shared memory

thurst library
