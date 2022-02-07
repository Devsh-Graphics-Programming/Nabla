// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_INCLUDE_HANDLER_H_INCLUDED__
#define __NBL_ASSET_I_INCLUDE_HANDLER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/utils/IBuiltinIncludeLoader.h"
#include "nbl/system/path.h"
namespace nbl
{
namespace asset
{
class IIncludeHandler : public core::IReferenceCounted
{
public:
    static constexpr const char* BUILTIN_PREFIX = "nbl/builtin/";
    static bool isBuiltinPath(const system::path& _p)
    {
        const size_t prefix_len = strlen(BUILTIN_PREFIX);
        return _p.string().compare(0u, prefix_len, BUILTIN_PREFIX) == 0;
    }

protected:
    virtual ~IIncludeHandler() = default;

public:
    virtual std::string getIncludeStandard(const system::path& _path) const = 0;
    virtual std::string getIncludeRelative(const system::path& _path, const system::path& _workingDirectory) const = 0;

    virtual void addBuiltinIncludeLoader(core::smart_refctd_ptr<IBuiltinIncludeLoader>&& _inclLoader) = 0;
};

}
}

#endif
