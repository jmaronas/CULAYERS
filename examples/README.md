# Examples of CULAYERS

This folder holds one example to demonstrate the easy usage of culayers. Just put the file inside examples at the same level of the necessary .h (see general instructions). Execute make and the compile the file by executing. Example:

```
git clone https://github.com/jmaronas/CULAYERS.git
cd CULAYERS
make
make clean
mv ./examples/* .
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64/":"./"
export PATH=$PATH:/usr/local/cuda-8.0/bin/
nvcc ForwardNNet.cpp -o executable -L./ -lculayers
./executable
```
Follow the commented code and check the output. 
