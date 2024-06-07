#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_


#include "nbl/asset/utils/CGLSLCompiler.h" // asset::CGLSLCompiler::E_SPIRV_VERSION

#include "nbl/asset/IImage.h"
#include "nbl/asset/IRenderpass.h"

#include <type_traits>


namespace nbl::video
{

// Struct is populated with Nabla Core Profile Limit Minimums
struct SPhysicalDeviceLimits
{
	#include "nbl/video/test_device_limits.h"
};

} // nbl::video

#endif
