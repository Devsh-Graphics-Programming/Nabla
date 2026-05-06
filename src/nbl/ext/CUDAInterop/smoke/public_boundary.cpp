#include "nabla.h"
#include "nbl/ext/CUDAInterop/CUDAInterop.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#error "Default Nabla consumers must not get the CUDA opt-in define."
#endif

#ifdef CUDA_VERSION
#error "Default Nabla consumers must not include CUDA SDK headers."
#endif

int main()
{
	return 0;
}
