#ifndef __IRR_C_BUILTIN_INCLUDE_HANDLER_H_INCLUDED__
#define __IRR_C_BUILTIN_INCLUDE_HANDLER_H_INCLUDED__

#include "CObjectCache.h"
#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr { namespace asset
{

class CBuiltinIncludeHandler
{
    CBuiltinIncludeHandler() = delete;

    using LoadersContainer = core::CMultiObjectCache<std::string, IBuiltinIncludeLoader>;
    static LoadersContainer m_loaders;

public:
    static std::string getBuiltinInclude(const std::string& _path)
    {
        auto capableLoadersRng = m_loaders.findRange(_path.substr(0u, _path.find_last_of('/')));
        for (auto loaderItr = capableLoadersRng.first; loaderItr != capableLoadersRng.second; ++loaderItr)
            if (loaderItr->second->canLoad(_path))
                return loaderItr->second->getBuiltinInclude(_path);
        return {};
    }

    static void addBuiltinLoader(IBuiltinIncludeLoader* _loader)
    {
        if (!_loader)
            return;

        m_loaders.insert(_loader->getVirtualDirectoryName(), _loader);
    }
};

}}

#endif//__IRR_C_BUILTIN_INCLUDE_HANDLER_H_INCLUDED__