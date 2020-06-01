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
		virtual ~IGLSLEmbeddedIncludeLoader() = default;

		inline core::vector<std::pair<std::regex,HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
		{
			//auto pattern = std::regex_replace(std::string(getVirtualDirectoryName()), std::regex{"\/"}, "\\/");
			std::string pattern(getVirtualDirectoryName());
			pattern += ".*";
			HandleFunc_t tmp = [this](const std::string& _name) -> std::string {return getFromDiskOrEmbedding(_name);};
			return {{std::regex{pattern},std::move(tmp)}};
		}
		
		
		io::IFileSystem* fs;

	public:
		IGLSLEmbeddedIncludeLoader(io::IFileSystem* filesystem) : fs(filesystem) {}

		//
		inline std::string getFromDiskOrEmbedding(const std::string& _name) const
		{
			auto path = "irr/builtin/" + _name;
			return reinterpret_cast<const char*>(fs->loadBuiltinData(path)->getPointer());
		}
};

}
}

#endif//__IRR_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
