#include "nbl/system/ILogger.h"

namespace nbl::system
{
core::bitflag<ILogger::E_LOG_LEVEL> ILogger::defaultLogMask()
{
#ifdef _NBL_DEBUG
    return core::bitflag(ELL_DEBUG) | ELL_INFO | ELL_WARNING | ELL_PERFORMANCE | ELL_ERROR;
#elif defined(_NBL_RELWITHDEBINFO)
    return core::bitflag(ELL_INFO) | ELL_WARNING | ELL_PERFORMANCE | ELL_ERROR;
#else
    return core::bitflag(ELL_WARNING) | ELL_PERFORMANCE | ELL_ERROR;
#endif
}

}