// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_INCLUDE_HANDLER_H_INCLUDED__
#define __NBL_ASSET_I_INCLUDE_HANDLER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/IBuiltinIncludeLoader.h"

namespace nbl
{
namespace asset
{

class IIncludeHandler : public core::IReferenceCounted
{
	public:
		_NBL_STATIC_INLINE_CONSTEXPR char* BUILTIN_PREFIX = "nbl/builtin/";
		static bool isBuiltinPath(const std::string& _p)
		{
			const size_t prefix_len = strlen(BUILTIN_PREFIX);
			return _p.compare(0u, prefix_len, BUILTIN_PREFIX)==0;
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

#endif
