#include "nbl/system/ILogger.h"

namespace nbl::system
{
std::underlying_type_t<ILogger::E_LOG_LEVEL> ILogger::defaultLogMask()
{
#ifdef _NBL_DEBUG
	return ELL_DEBUG | ELL_INFO | ELL_WARNING |	ELL_PERFORMANCE | ELL_ERROR;
#elif defined(_NBL_RELWITHDEBINFO)
	return ELL_INFO | ELL_WARNING | ELL_PERFORMANCE | ELL_ERROR;
#else
	return ELL_WARNING | ELL_PERFORMANCE | ELL_ERROR;
#endif

}

}