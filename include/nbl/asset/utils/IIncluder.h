// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_INCLUDER_H_INCLUDED__
#define __NBL_ASSET_I_INCLUDER_H_INCLUDED__

#include <string>

#include "nbl/system/path.h"
#include "nbl/core/declarations.h"

namespace nbl
{
namespace asset
{
class IIncluder : public core::IReferenceCounted
{
protected:
    core::vector<std::filesystem::path> m_searchDirectories;

    virtual ~IIncluder() = default;

public:
    IIncluder()
        : m_searchDirectories{""} {}

    virtual void addSearchDirectory(const system::path& _searchDir) { m_searchDirectories.push_back(_searchDir); }

    std::string getIncludeStandard(const system::path& _path) const
    {
        for(const system::path& searchDir : m_searchDirectories)
        {
            system::path path = searchDir;
            path += _path;
            if(std::filesystem::exists(path))
                path = std::filesystem::canonical(path).string();
            std::string res = getInclude_internal(path);
            if(!res.empty())
                return res;
        }
        return {};
    }
    std::string getIncludeRelative(const system::path& _path, const system::path& _workingDir) const
    {
        system::path path = _workingDir;
        if(!_workingDir.empty() && *_workingDir.string().rbegin() != '/')
            path += "/";
        path += _path;
        if(std::filesystem::exists(path))
            path = std::filesystem::canonical(path);
        return getInclude_internal(path);
    }

protected:
    //! Always gets absolute path
    virtual std::string getInclude_internal(const system::path& _path) const = 0;
};

}
}

#endif