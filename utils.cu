#include <string.h>
#include <stdio.h>
#include "utils.h"
#include "utils_gpu_env.h":
#include "gpu_env.h"
/////////////////////////////
void printDebug(float* p,const char* str, int row,int col,gpu_env gpu_tensor_op)
{
float* aux = (float*)malloc(row*col*sizeof(float));

gpu_tensor_op.copy_data(aux,p,FROMGPU,row*col*sizeof(float));
printf("%s\n",str);
for (int i=0;i<row;i++)
{
for (int j=0;j<col;j++)
{
        printf("%f ",aux[i*col+j]);
}
printf("\n");
}
free(aux);
}


