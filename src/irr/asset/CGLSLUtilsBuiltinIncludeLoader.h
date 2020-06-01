#ifndef __IRR_ASSET_C_GLSL_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_ASSET_C_GLSL_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IGLSLEmbeddedIncludeLoader.h"

namespace irr
{
namespace asset
{    

class CGLSLUtilsBuiltinIncludeLoader : public IGLSLEmbeddedIncludeLoader
{
    public:
        using IGLSLEmbeddedIncludeLoader::IGLSLEmbeddedIncludeLoader;

        const char* getVirtualDirectoryName() const override { return "glsl/utils/"; }
};

}
}

#endif//__IRR_ASSET_C_GLSL_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__