#include "CCUDAHandler.h"


#ifdef _IRR_COMPILE_WITH_CUDA_

namespace irr
{
namespace cuda
{

CCUDAHandler::CUDA CCUDAHandler::cuda;
CCUDAHandler::NVRTC CCUDAHandler::nvrtc;

core::vector<CCUDAHandler::Device> CCUDAHandler::devices;

}
}

#endif // _IRR_COMPILE_WITH_CUDA_
