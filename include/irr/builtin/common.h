#ifndef _IRR_BUILTIN_COMMON_H_INCLUDED_
#define _IRR_BUILTIN_COMMON_H_INCLUDED_

#include "BuildConfigOptions.h"

#ifndef _IRR_EMBED_BUILTIN_RESOURCES_
// will only be available in the app or library using irrlicht
#if defined(_IRR_STATIC_LIB_) || !defined(IRRLICHT_EXPORTS)

#define _IRR_BUILTIN_PATH_AVAILABLE
namespace irr
{
namespace builtin
{

inline constexpr char* getBuiltinResourcesCommonHeaderPath()
{
    return __FILE__;
}

}
}
#endif
#endif

#endif