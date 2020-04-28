#include "COpenCLHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENCL_
namespace irr
{
namespace ocl
{

bool COpenCLHandler::alreadyEnumeratedPlatforms = false;

core::vector<COpenCLHandler::SOpenCLPlatformInfo> COpenCLHandler::platformInformation;


COpenCLHandler::OpenCL COpenCLHandler::ocl;
COpenCLHandler::OpenCLExtensions COpenCLHandler::ocl_ext;

}
}


#endif // _IRR_COMPILE_WITH_OPENCL_
