#ifndef _UTILSGPUENV_
#define _UTILSGPUENV_

#include <stdint.h>

#define ACT_LIN 0
#define ACT_RLU 1
#define ACT_SIG 2
#define ACT_ELU 3
#define ACT_SOF 10

typedef struct
{
int batch;//only for convolutional
int featureMap;//only for convolutional
int row;
int col;
}tensor_gpu_specs;

typedef enum
{
CE,
SSE
}loss_type;

typedef enum
{
TOGPU,
FROMGPU,
GPU,
CPU
}tr_type;


typedef struct
{
uint32_t warpSize;
uint32_t maxThreadPerBlock;
uint32_t multiCores;
uint32_t totalGlobMem;
uint32_t sharedMemPerBlock;
uint32_t totalConstMem;
uint32_t clockRate;
}gpu_specs;


#endif
