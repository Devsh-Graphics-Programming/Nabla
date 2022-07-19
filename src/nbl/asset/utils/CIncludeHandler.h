// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_INCLUDE_HANDLER_H_INCLUDED__
#define __NBL_ASSET_C_INCLUDE_HANDLER_H_INCLUDED__

#include "nbl/asset/utils/IIncludeHandler.h"

#include "CFilesystemIncluder.h"
#include "CBuiltinIncluder.h"

namespace nbl
{
namespace asset
{

class CIncludeHandler : public IIncludeHandler
{
    using SmartRefctIncluderPtr = core::smart_refctd_ptr<IIncluder>;

    core::vector<SmartRefctIncluderPtr> m_includers;

    enum E_INCLUDER_INDEX
    {
        EII_FILESYSTEM = 0,
        EII_BUILTIN = 1
    };

public:
    CIncludeHandler(system::ISystem* _system)
    {
        // TODO It has to be reworked in the future
        m_includers.emplace_back(core::make_smart_refctd_ptr<CFilesystemIncluder>(_system));
        m_includers.emplace_back(core::make_smart_refctd_ptr<CBuiltinIncluder>(_system));
    }

    std::string getIncludeStandard(const system::path& _path) const override
    {
        return getIncluderDependentOnPath(_path)->getIncludeStandard(_path);
    }

    std::string getIncludeRelative(const system::path& _path, const system::path& _workingDirectory) const override
    {
        return getIncluderDependentOnPath(_path)->getIncludeRelative(_path, _workingDirectory);
    }

    void addBuiltinIncludeLoader(core::smart_refctd_ptr<IBuiltinIncludeLoader>&& _inclLoader) override
    {
        static_cast<CBuiltinIncluder*>(m_includers[EII_BUILTIN].get())->addBuiltinLoader(std::move(_inclLoader));
    }

private:
    const IIncluder* getIncluderDependentOnPath(const system::path& _path) const
    {
        return (isBuiltinPath(_path) ? m_includers[EII_BUILTIN].get() : m_includers[EII_FILESYSTEM].get());
    }
};

}
}

#endif
