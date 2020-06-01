#ifndef __IRR_ASSET_I_GLSL_EMBEDDED_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_ASSET_I_GLSL_EMBEDDED_INCLUDE_LOADER_H_INCLUDED__

#include "irr/system/system.h"
#include "IFileSystem.h"

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr
{
namespace asset
{

class IGLSLEmbeddedIncludeLoader : public IBuiltinIncludeLoader
{
	protected:
		IGLSLEmbeddedIncludeLoader(io::IFileSystem* filesystem) : fs(filesystem) {}
		virtual ~IGLSLEmbeddedIncludeLoader() = default;

		inline core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
		{
			str::string pattern(getVirtualDirectoryName());
			return {{ std::regex{},&getFromDiskOrEmbedding }};
		}
		
		
		io::IFileSystem* fs;

	public:
		//
		inline std::string getFromDiskOrEmbedding(const std::string&) const
		{
			auto path = "irr/builtin/" + _name;
			return reinterpret_cast<const char*>(fs->loadBuiltinData(path)->getPointer());
		}
};

}
}

#endif//__IRR_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
