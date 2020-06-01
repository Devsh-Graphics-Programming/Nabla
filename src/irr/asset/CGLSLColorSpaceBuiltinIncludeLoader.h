#ifndef __IRR_ASSET_C_GLSL_COLOR_SPACE_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_ASSET_C_GLSL_COLOR_SPACE_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IGLSLEmbeddedIncludeLoader.h"

namespace irr
{
namespace asset
{    

class CGLSLColorSpaceBuiltinIncludeLoader : public IGLSLEmbeddedIncludeLoader
{
    public:
        const char* getVirtualDirectoryName() const override { return "glsl/colorspace/"; }
};

}
}

#endif//__IRR_ASSET_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__