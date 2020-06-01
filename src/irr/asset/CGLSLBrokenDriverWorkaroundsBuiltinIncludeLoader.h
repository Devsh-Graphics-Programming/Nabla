#ifndef __IRR_ASSET_C_GLSL_BROKEN_DRIVER_WORKAROUNDS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_ASSET_C_GLSL_BROKEN_DRIVER_WORKAROUNDS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IGLSLEmbeddedIncludeLoader.h"

namespace irr 
{
namespace asset
{

class CGLSLBrokenDriverWorkaroundsBuiltinIncludeLoader : public IGLSLEmbeddedIncludeLoader
{
	public:
		using IGLSLEmbeddedIncludeLoader::IGLSLEmbeddedIncludeLoader;

		const char* getVirtualDirectoryName() const override { return "glsl/broken_driver_workarounds/"; }
};

}
}

#endif // __IRR_C_GLSL_BROKEN_DRIVER_WORKAROUNDS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__