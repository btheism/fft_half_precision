#include<exception>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        throw std::runtime_error("GPUassert"+std::string(cudaGetErrorString(code))+std::string(file));
        //printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
    /*
    else
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
    */
}

