#ifndef __IRR_C_BUILTIN_INCLUDER_H_INCLUDED__
#define __IRR_C_BUILTIN_INCLUDER_H_INCLUDED__

#include "irr/asset/IIncluder.h"
#include "CObjectCache.h"
#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr { namespace asset
{

class CBuiltinIncluder : public IIncluder
{
    using LoadersContainer = core::CMultiObjectCache<std::string, IBuiltinIncludeLoader>;
    LoadersContainer m_loaders;

public:
    CBuiltinIncluder()
    {
        m_searchDirectories.emplace_back("/");
    }

    //! No-op, cannot add search dirs to includer of builtins
    void addSearchDirectory(const std::string& _searchDir) override {}

    std::string getInclude_internal(const std::string& _path) const override
    {
        std::string res;
        auto capableLoadersRng = m_loaders.findRange(_path.substr(0, _path.find_last_of('/')));
        for (auto loaderItr = capableLoadersRng.first; loaderItr != capableLoadersRng.second; ++loaderItr)
            if (!(res = loaderItr->second->getBuiltinInclude(_path)).empty())
                return res;
        return {};
    }

    void addBuiltinLoader(IBuiltinIncludeLoader* _loader)
    {
        using namespace std::string_literals;
        if (!_loader)
            return;

        m_loaders.insert("/irr/builtin/"s + _loader->getVirtualDirectoryName(), _loader);
    }
};

}}

#endif//__IRR_C_BUILTIN_INCLUDER_H_INCLUDED__