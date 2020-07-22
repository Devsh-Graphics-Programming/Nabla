#ifndef __IRR_ASSET_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_ASSET_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include <functional>
#include <regex>


#include "irr/core/IReferenceCounted.h"

namespace irr
{
namespace asset
{

class IBuiltinIncludeLoader : public core::IReferenceCounted
{
	protected:
		using HandleFunc_t = std::function<std::string(const std::string&)>;

		virtual core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const = 0;

	public:
		virtual ~IBuiltinIncludeLoader() = default;

		//! @param _name must be path relative to /irr/builtin/
		virtual std::string getBuiltinInclude(const std::string& _name) const
		{
			core::vector<std::pair<std::regex, HandleFunc_t>> builtinNames = getBuiltinNamesToFunctionMapping();

			for (const auto& pattern : builtinNames)
				if (std::regex_match(_name, pattern.first))
					return pattern.second(_name);

			return {};
		}

		//! @returns Path relative to /irr/builtin/
		virtual const char* getVirtualDirectoryName() const = 0;
};

}
}

#endif//__IRR_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
