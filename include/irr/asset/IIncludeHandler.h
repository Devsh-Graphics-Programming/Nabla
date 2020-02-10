#ifndef __IRR_I_INCLUDE_HANDLER_H_INCLUDED__
#define __IRR_I_INCLUDE_HANDLER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr
{
namespace asset
{

class IIncludeHandler : public core::IReferenceCounted
{
	public:
		static constexpr const char* BUILTIN_PREFIX = "/irr/builtin/";
		static bool isBuiltinPath(const std::string _p) {
				size_t prefix_len = strlen(BUILTIN_PREFIX);
				return _p.compare(0u, prefix_len, BUILTIN_PREFIX)==0 || _p.compare(0u, prefix_len-1u, BUILTIN_PREFIX+1)==0;
		}

	protected:
		virtual ~IIncludeHandler() = default;

	public:
		virtual std::string getIncludeStandard(const std::string& _path) const = 0;
		virtual std::string getIncludeRelative(const std::string& _path, const std::string& _workingDirectory) const = 0;

		virtual void addBuiltinIncludeLoader(core::smart_refctd_ptr<IBuiltinIncludeLoader>&& _inclLoader) = 0;
};

}
}

#endif//__IRR_I_INCLUDE_HANDLER_H_INCLUDED__
