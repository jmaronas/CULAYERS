# LIBRARY DESCRIPTION

This file describes all the library headers available and their parameters. culayers is organized as a class which serves as an interface to operations on gpu, gpu info... The good point of culayers is that each class creates its own handles to gpu, and thus, you can use different gpu_env (gpu enviroments) to be runned in a multithread application running on different GPUs, does culayers can be used in parallel application without any additional effort. Here I descript the public methods and variables. Check gpu_env.h for the private ones.

## Typedefs:
### typedef struct tensor_gpu_specs
This typedef is used to tell culayers how to manage the different linear arrays. It has four attributes. As I do not handle convolutions for the moment I will describe the two needed:

row: number of rows of your array

col: number of columns of your array
### Directives

ACT_LIN: apply linear activation

ACT_RLU: apply relu activation

ACT_SIG: apply sigmoid activation

ACT_ELU: apply elu activation

ACT_SOF: apply softmax function

### typedef struct gpu_specs
This typedef es filled with information of the gpu used. You can access it through the global variable ```extern gpu_specs gpuI;``` which is available when including utils.h. TODO: describe the parameters

### typedef enum loss_type
Typedef that describes different cost functions:

CE: cross entropy

SSE: sum of squared errors

### typedef enum tr_type
This typedef is used to specify memory transfers. It has the following enumerations:

TOGPU: transfer TO gpu (from cpu)

TOCPU: transfer TO cpu (from gpu)

GPU: transfer between gpu

CPU: transfer between cpu. Note that this can be done in normal C++ code.

## Methods 
### Constructor:
This function is the constructor of the class. By default gpu_env randomness is initialized with seed=1. Example
```
operator gpu_env;
```
From now on operator will be refered to our gpu enviroment

### Change seed: void set_seed(int seed)
This functions reinitialized the random generator to the given seed. Example
```
operator.set_seed(2);//now culayers generate random numbers with that seed
```
### GPU info: void gpu_info(int selected_gpu)
Get information on the gpu given by selected_gpu and select that gpu as the one to be used. GPU counter starts from 0 till n-1 where n is the natural number representing all the GPUs available. Example:
```
operator.gpu_info(0);//Gets gpu info from the first gpu in the system and select it for use
```

## Memory Managment
culayers gives you an easy interface to create tensors in your GPU and your CPU, and the possibility to transfer them from/to the cpu.

### Create a tensor: float* makeTensor(int a, int b=1,int c=1, int d=1);
Creates a tensor of given shapes ```a*b*c*d```. You are responsible of the correct interpretation of this tensor dimension. Tensors follow the standard C-convention row-major order. If a function uses column-major order you do not have to care about that, culayers does that for you. For each returned tensor you should associate a tensor_gpu_op structure that is used by culayers to manage the array returned by this function. One dimensional arrays are tipically more efficient that other complex structures and I rely on the functions provided by NVIDIA to do that. This is important to fetch operators and can considerably reduce the number of access to GPU memory.

So as example for create a matrix of 100 rows and 784 columns you can both do:

```
// This three options allocate the same amount of memory
float gpuATensor = operator.makeTensor(100,784);
float gpuATensor = operator.makeTensor(100*784);
float gpuATensor = operator.makeTensor(784,100);
tensor_gpu_specs gpuASpecs; 
gpuASpecs.row=100;gpuBSpecs.row=784;//this is what culayer use to correctly use the allocate memory
```
### Destroy Tensor: void destroyTensor(float* p);
Destroy the given tensor p. Example:
```
float gpuATensor = operator.makeTensor(100,784);
operator.destroyTensor(gpuATensor);
```

### copy data between different memory locations: copy_data(float* cpu, float* gpu, tr_type t, size_t size);
Copy/transfer data between different arrays. Arrays can be either in cpu or gpu. Tipically this function is used to transfer between gpu and cpu but can be used to transfer between gpu and gpu and between cpu and cpu. Memory on the pointers must be allocated when calling this function Description on arguments:

float* cpu: cpu float pointer (call to malloc)

float* gpu: gpu float pointer (call to makeTensor)

tr_type t: typedef enum (check above for its description)

size_t size: total bytes to transfer (tipically n_floats\*sizeof(float))

There is a special case, when copying one gpu tensor to other gpu tensor the argument in cpu is the destination and the argument in gpu is the origin. Examples:

```
//asume you have allocate the memory for (100,784) matrix in cpuA (loaded from disk as example) and you want to copy to gpuA
operator.copy_data(cpuA,gpuA,TOGPU,100*784*sizeof(float));//copy the contents in cpuA to gpuA

//you make some operations and one to store the result of gpuC in cpuC to display (maybe matrix (100,10))
operator.copy_data(cpuC,gpuC,FROMGPU,100*10*sizeof(float));//copy the contents in cpuA to gpuA

//you want to perform some transfer learning and one to copy your ImageNet model into a new model.
operator.copy_data(gpuMyModel,gpuImageNet,GPU,100*512*sizeof(float));//copy the contents in gpuImageNet to gpuMyModel

```
### Set scalar value: set_sc(float* gpu, float sc, tensor_gpu_specs* sp);
Set all the tensor gpu with the scalar value sc. sp stores the information of the tensor, as explained. Example
```
float gpuATensor = operator.makeTensor(784,100);
tensor_gpu_specs gpuASpecs; 
operator.set_sc(gpuA,0.5,&gpuASpecs);//set gpuA with 0.5 value
```

## Neural Networks:
This functions perform tipical neural networks operations.

### Activation Function: void activation(float* E,float* N, int f,tensor_gpu_specs* sp)
This functions apply the activation function given by int f. Int f is a defined directive (check above) which represent the activation to perform. N is where the result of the operation is placed and E is the source tensor where the activation is applied. Sp is the tensor_gpu_spec struct whith can be either the one for N or for E, as this operation is apply element-wise, and thus assumes both tensor have similar shape. The softmax function is the only one that expect a 2-D matrix as input. Example:

```
operator.activation(gpuOutput,gpuIn,RLU,specsOutput); //apply relu to gpuIn and saves into gpuOutput
```

### Derivative of Activation: void dactivation(float* E,float* N,float* D, int f, tensor_gpu_specs* sp)
 Same as activation but returns the derivative of that activation function given by f. D is the tensor where the output is placed, E is the input tensor to the activation and N is the output of that activation. This is needed to perform the derivative depending on the type of activation. The derivative is evaluated on the input, as this functions are used to perform back propagation. Example:
 
```
// assume gpuC=relu(X*gpuB). You store derivative in dTensor. gpuB are paremters and X is the data, gpuC is the result. 
operator.dactivation(gpuB,gpuC,dTensor,RLU,&gpuSpecs)
```

### Compute a loss function: void compute_loss(float* T, float* N,loss_type t,tensor_gpu_specs* gsp,double* ent, double* cerr);
This function computes a loss function of a given tensor. T is the tensor with the targets and N is the tensor with your predictions. loss_type t is a enumerate type which defines which type of loss to apply (check above). gsp is the gpus_specs of any of the tensors. ent is a double variable where the total cross entropy error is placed and cerr is the number of errors. T is a one_hot vector. Example:

```
//gpuN is output of softmax, gpuT is the target vector
double errors;
double entropy;
operator.compute_loss(gpuT,gpuN,CE,gpuSpecs,entropy,errors);
```

## Linear Algebra
This section describes tipical linear algebra operations.

### Matrix Product: void matMul(float* a,float* b,float* c, tensor_gpu_specs* sA,tensor_gpu_specs* sB, tensor_gpu_specs* sC, int acc, int tA, int tB); 
Performs a matrix product between a and b and store the resultd on c. tensor_gpu_specs is the specification for each different tensor. If the operation is invalid (for example you try to multiply two matrix but the dimensions are not correct) the library raises and error.

acc{0,1}= if 1 accumulates the results if 0 sets the result in c

tA{0,1}= if 1 transposes matrix a if 0 do not tranpose

tB{0,1}= if 1 transposes matrix b if 0 do not transpose

Example:
```
float AT = operator.makeTensor(100,784);
float BT = operator.makeTensor(784,1000);
float CT = operator.makeTensor(100,1000);
tensor_gpu_specs ASpec,BSpec,CSpec; 
ASpec.row=100;ASpec.col=784;
BSpec.row=784;BSpec.col=1000;
CSpec.row=100;CSpec.col=1000;
operator.matMul(AT,BT,CT,ASpec,BSpec,CSpec,0,0,0)
```
Transposition operations or accumulations are provided for example for performing the backward in a neural network of a layer that is connected to several layers.

### Scalar operation by matrix: void scMat(float* mat_o,float* mat_i, tensor_gpu_specs* sA, float sc ,int op,int acc);
This function multiplies/sum the scalar sc with the matrix mat_i and store the result in matrix mat_o. acc denotes wether to accumulate or not the result in mat_o and op is a scalar indicating wether you perform a multiplication or a sum. 

op{0,1}: 0 sum, 1 multiplication

### Scalar operation by vector: void scVec(float* vec_o,float* vec_i, tensor_gpu_specs* sA, float sc ,int op,int acc);
Same as scalar operation by matrix but vec_i and vec_o represent vectors. 

op{0,1}: 0 sum, 1 multiplication

## Element-wise Operations
This functions perform element-wise operations on different structures.

### Element-wise operation between matrix: void mat_elwise_mat(float* A, float* B, float* C,tensor_gpu_specs* sA,tensor_gpu_specs* sB,tensor_gpu_specs* sC,int op, int acc, int tA, int tB,float sca=1.0, float scb=1.0);
Perform element wise operation between to matrixes with added functionality.

A: first matrix

B: second matrix

C: place the result

sA,sB,sC: specs of the matrixes

acc{0,1}: 0 set result in C,  1 accumulate result in C.

op{0,1}: 1 multiply 0 sum

tA,tB{0,1}: 0 do not transpose 1 transpose matrix

sca,scb: matrixes A and B are multiplied by this scalar.

### Compare two tensors: int tensor_equal(float* A, float* B, tensor_gpu_specs* sA);
This function returns 1 if both tensors are equal and 0 if not.

### Element-wise operation between two vectors: void vec_elwise_vec(float* A, float* B, float* C,tensor_gpu_specs* sA,int op, int acc,float sca=1.0, float scb=1.0);
Same as element-wise operation between matrix but A and B are vectors. Note that one could use above function specifying in sA or sB only a value for rows. However, this function uses optimized functions for vector operations. Note that transposition here does not make sense. Vector should be Row major vectors.

op{0,1}: 1 multiply 0 sum

acc{0,1}: 0 set result in C,  1 accumulate result in C.

### Element-wise operation between vector and matrix: void gpu_env::mat_elwise_vec(float* mat_o,float* mat, float* vec, tensor_gpu_specs* sA,int op,int acc,float sca, float scb, int rdim)
This function performs an operation of a vector with a given dimension of the matrix. For example you can sum a vector to each row of a matrix. Vector and matrixes can be multiplied by scalars

mat_o: matrix outpu

mat: matrix input

vec: vector

op{0,1,2}:2 division 1 multiply 0 sum

acc{0,1}:  0 set result in mat_o,  1 accumulate result in mat_o.

sca,scb: scalar to multiply matrix and vector respectively

r_dim{0,1}: if 1 operates by columns, if 0 operates by rows.

### Element-wise operation in a matrix: void mat_inplace_mat(float* o, float* i,tensor_gpu_specs* si,int op, int acc);
This function performs elemnt-wise operation in a matrix for those operations that only require one input tensor. For example computing the element-wise square root. O is the output tensor and i is the input tensor, with si denoting its specs.

op{0}: square root element wise

acc{0,1}: 0 no accumulate, 1 accumulate 

## Reduction Operators
This section explains reduction operators on different tensors.

### Sum by column: void col_sum(float* A, tensor_gpu_specs* sA,float* B);
Sum columns of matrix A and store in B. A should be a matrix and we place in vector B the sum by columns of matrix A. This means that vector B should have the same elements as sA.col

### Sum by row: void row_sum(float* A, tensor_gpu_specs* sA,float* B);
Sum rows of matrix A and store in B. A should be a matrix and we place in vector B the sum by rows of matrix A. This means that vector B should have the same elements as sA.row

### Compute maximum of a row: int row_max(float* A, tensor_gpu_specs* sA,int ind );
Returns the maximum value of matrix A in the row specified by ind. This can serve as to make predictions in neural network classifier.

### Sum of absolute values: void sum_abs(float* p,tensor_gpu_specs* gsp,float* acc);
Return the sum of absolute values of tensor p in float* acc. Memory should be allocated for acc.

### Reduce dimension: void reduce_operator(float* p,tensor_gpu_specs* gsp,float* acc);
Sum all the elements fo array p and store result in acc. Same as above but without computing absolute value.

## Random Operators
Several functions for adding randomness to your tensors, for example if you want to add dropout or sample from a generative adversarial network.

### void add_noise(float* vec, float* rand_vec,float noiser, tensor_gpu_specs* sp);
Add noise the noise store in rand_vec to to tensor vec using the noise ratio noiser (that is, the probability that we add noise to a point in the tensor). This is used to add gaussian noise, for example. You sample from random_number_host_gaussian and use this function to add that gaussian noise with element-wise-probability noiser.

### void random_number_host_gaussian(float* rand_vec,tensor_gpu_specs* sp,float mean,float std);
Samples from a Gaussian distribution with mean and std given by parameters and store in rand_vec float pointer. 

### void random_number_host_binary(float* rand_vec,tensor_gpu_specs* sp,float p);
Sample from a Bernouilli  distribution with probability p. We sample from a uniform distribution and then apply a mask. We then return the mask (which is fill with 1 and 0 depending on the probability p). This mask can be multiplied row-wise with a tensor to apply a dropout layer.

## DEBUG FUNCTIONS
The next function and types are useful for information and debug.

### Print a tensor: void printDebug(float* p,const char* str, int row,int col,gpu_env gpu_tensor_op)
Gpu tensors cannot be printed with a for loop. We must copy them to host and the print. For that reason I have done this function for easily debugging of your tensors.

p: gpu tensor to pring

str: a string representing a message you want to display

row: number of rows

col: number of column

gpu_tensor_op: the enviroment used with that tensor
