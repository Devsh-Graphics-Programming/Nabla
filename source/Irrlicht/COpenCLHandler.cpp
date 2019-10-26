#include "COpenCLHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENCL_
namespace irr
{
namespace ocl
{


bool COpenCLHandler::alreadyEnumeratedPlatforms = false;
#if defined(_IRR_WINDOWS_API_)
clGetGLContextInfoKHR_fn COpenCLHandler::pClGetGLContextInfoKHR = NULL;
#endif // defined
cl_platform_id COpenCLHandler::platforms[IRR_maxOCL_PLATFORMS];
cl_uint COpenCLHandler::actualPlatformCount = 0;
COpenCLHandler::SOpenCLPlatformInfo COpenCLHandler::platformInformation[IRR_maxOCL_PLATFORMS];


}
}


#endif // _IRR_COMPILE_WITH_OPENCL_
