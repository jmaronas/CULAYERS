#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "gpu_env.h"
#include "utils.h"

void fill_with_random(float* p, int n)
{
for (int i=0;i<n;i++)
	p[i]=(float)rand()/(float)RAND_MAX;
}

int main(int argc, char** argv)
{

gpu_env gpu_operator;//create our enviroment. By default it select gpu device 0. You can create an instance following gpu_env gpu_operator(gpu_id); to select other gpu.

// You can get information on your gpu system. This also set the gpu to the one pass as argument.
gpu_operator.gpu_info(0);

//Assume we want to perform a forward over a 1 hidden neural network trained on MNIST. We perform 3 basic operations. We multiply matrix, add bias and apply non-linearities. We will simulate everything with random numbers. For printint we assume MNIST has now 10 dimensions, our hidden layer 5 and our output layer 2 (ie binary classification). We have a batch of four samples

float* X = gpu_operator.makeTensor(4,10);//create a 4x10 matrix on GPU
float* W1 = gpu_operator.makeTensor(10,5);//our projection matrix to hidden layer
float* b1 = gpu_operator.makeTensor(5);//bias of that layer
float* W2 = gpu_operator.makeTensor(5,2);//our projection matrix to output layer
float* b2 = gpu_operator.makeTensor(2);//bias for output layer

//you fill your data structure associated with tensors
tensor_gpu_specs sX,sW1,sW2,sb1,sb2;
sX.row=4;sX.col=10;
sW1.row=10;sW1.col=5;
sb1.row=5;
sW2.row=5;sW2.col=2;
sb2.row=2;

//We now fill our data with some random stuff

//Use C-major order, arguments to the lib changes to R-major order if needed. Specified in docs
float cpuX[4*10],cpuW1[10*5],cpub1[5],cpuW2[5*2],cpub2[2];
srand(1);//fill with randomness
fill_with_random(cpuX,40);
fill_with_random(cpuW1,50);
fill_with_random(cpuW2,10);
fill_with_random(cpub2,2);
fill_with_random(cpub1,5);

//now copy your data to gpu
printf("Copying data to gpu...\n");
gpu_operator.copy_data(cpuX, X,TOGPU, 40*sizeof(float));
gpu_operator.copy_data(cpuW1, W1,TOGPU, 50*sizeof(float));
gpu_operator.copy_data(cpuW2, W2,TOGPU, 10*sizeof(float));
gpu_operator.copy_data(cpub1, b1,TOGPU, 5*sizeof(float));
gpu_operator.copy_data(cpub2, b2,TOGPU, 2*sizeof(float));
printf("success\n");

//first we try linear algebra operator. We need to project X to the hidden layer.
printDebug(X,"\nMatrix X",4,10,gpu_operator);
printDebug(W1,"\nMatrix W1",10,5,gpu_operator);
printDebug(b1,"\nBias b1",1,5,gpu_operator);
//to store intermediate results in tensor h_pre
float* hpre = gpu_operator.makeTensor(4,5);//create a 5x3 matrix on GPU
tensor_gpu_specs shpre;
shpre.row=4;shpre.col=5;
float* hpos = gpu_operator.makeTensor(4,5);//I create this one to show for example dropout acting in train mode and test mode
tensor_gpu_specs shpos;
shpos.row=4;shpos.col=5;

//perform operation and display result
gpu_operator.matMul(X,W1,hpre, &sX,&sW1,&shpre, 0, 0, 0); //do not transpose and do not accumulate
printDebug(hpre,"\nMatrix hpre",4,5,gpu_operator);

//now sum the bias row-wise
gpu_operator.mat_elwise_vec(hpre,hpre,b1, &shpre,0,0,1.0, 1.0,1); //we can overwrite the input matrix.R_dim is 1 because we want to do the operation row-wise, ie, we operate by columns
printDebug(hpre,"\nMatrix post_bias",4,5,gpu_operator);

//no apply activation function, a sigmoid, again inplace
gpu_operator.activation(hpre,hpre,ACT_SIG,&shpre);
printDebug(hpre,"\nMatrix post activation",4,5,gpu_operator);

//now we can apply dropout, a tipical operation
//in train we apply it this way
//we first generate a binary mask
float* my_rand_vec=gpu_operator.makeTensor(4,5);
gpu_operator.random_number_host_binary(my_rand_vec,&shpre,0.5);

//now multiply the matrix element-wise
gpu_operator.mat_elwise_mat(hpre,my_rand_vec,hpos,&shpre,&shpre,&shpos,1,0,0,0,1,1);
printDebug(my_rand_vec,"\nDropout mask",4,5,gpu_operator);
printDebug(hpos,"\nApplied dropout in train mode",4,5,gpu_operator);

//in test we multiply by drop probability 0.5. We have several options here to combine the different operators implemented. A simple one is just multiply matrix by scalar
gpu_operator.scMat(hpre,hpre,&shpre,0.5,1,0);
printDebug(hpre,"\nApplied dropout in test mode",4,5,gpu_operator);

//now we project to the output layer
float* logit = gpu_operator.makeTensor(4,2);//to store the presoftmax
tensor_gpu_specs slogit; slogit.row=4;slogit.col=2;
gpu_operator.matMul(hpre,W2,logit, &shpre,&sW2,&slogit, 0, 0, 0); //do not transpose and do not accumulate
gpu_operator.mat_elwise_vec(logit,logit,b2, &slogit,0,0,1.0, 1.0,1); //we can overwrite the input matrix.R_dim is 1 because we want to do the operation row-wise, ie, we operate by columns

printDebug(W2,"\nW2",5,2,gpu_operator);
printDebug(b2,"\nb2",1,2,gpu_operator);
printDebug(logit,"\nLogit",4,2,gpu_operator);

//and now compute the softmax
gpu_operator.activation(logit,logit,ACT_SOF,&slogit);
printDebug(logit,"\nSoftmax output",4,2,gpu_operator);
float* sum_by_row=gpu_operator.makeTensor(4);
gpu_operator.row_sum(logit,&slogit,sum_by_row);
printDebug(sum_by_row,"\nSoftmax output sum by row",4,1,gpu_operator);

//lets create a invented target
float* T = gpu_operator.makeTensor(4,2);//bias for output layer
float cpuT[4*2]={1,0,0,1,0,1,0,1};
gpu_operator.copy_data(cpuT, T,TOGPU, 8*sizeof(float));
printDebug(T,"\nOne hot target vector",4,2,gpu_operator);

//now we compute the loss, we now that computing the cross entropy we have the number of correct samples
double ceent,cerr;
gpu_operator.compute_loss(T,logit,CE,&slogit,&ceent, &cerr);
printf("\nNumber of errors %f \t cross entropy %f\n",cerr,ceent);

return 0;
}
