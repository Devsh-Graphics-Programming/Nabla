// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __NBL_ASSET_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include <functional>
#include <regex>

#include "nbl/core/IReferenceCounted.h"

namespace nbl
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

    //! @param _name must be path relative to /nbl/builtin/
    virtual std::string getBuiltinInclude(const std::string& _name) const
    {
        core::vector<std::pair<std::regex, HandleFunc_t>> builtinNames = getBuiltinNamesToFunctionMapping();

        for(const auto& pattern : builtinNames)
            if(std::regex_match(_name, pattern.first))
            {
                auto a = pattern.second(_name);
                return a;
            }

        return {};
    }

    //! @returns Path relative to /nbl/builtin/
    virtual const char* getVirtualDirectoryName() const = 0;
};

}
}

#endif
