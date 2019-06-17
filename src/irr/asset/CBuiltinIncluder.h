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

    static void loaderGrab(IBuiltinIncludeLoader* _ldr) { _ldr->grab(); }
    static void loaderDrop(IBuiltinIncludeLoader* _ldr) { _ldr->drop(); }

public:
    CBuiltinIncluder() : m_loaders(&loaderGrab, &loaderDrop)
    {
        m_searchDirectories.emplace_back("/");
    }

    //! No-op, cannot add search dirs to includer of builtins
    void addSearchDirectory(const std::string& _searchDir) override {}

    void addBuiltinLoader(IBuiltinIncludeLoader* _loader)
    {
        using namespace std::string_literals;
        if (!_loader)
            return;

        m_loaders.insert("/irr/builtin/"s + _loader->getVirtualDirectoryName(), _loader);
    }

protected:
    std::string getInclude_internal(const std::string& _path) const override
    {
        const char* PREFIX = "/irr/builtin/";
        if (_path.compare(0, strlen(PREFIX), PREFIX) != 0)
            return {};

        std::string path = _path.substr(0, _path.find_last_of('/')+1);

        std::string res;
        while (path != PREFIX) // going up the directory tree
        {
            auto capableLoadersRng = m_loaders.findRange(path);
            for (auto loaderItr = capableLoadersRng.first; loaderItr != capableLoadersRng.second; ++loaderItr)
            {
                std::string relativePath = _path.substr(loaderItr->first.size(), std::string::npos); // builtin loaders take path relative to PREFIX
                if (!(res = loaderItr->second->getBuiltinInclude(relativePath)).empty())
                    return res;
            }
            path.back() = 'x'; // get rid of trailing slash...
            path.erase(path.find_last_of('/')+1, std::string::npos); // ...and find the one before
        }
        return {};
    }
};

}}

#endif//__IRR_C_BUILTIN_INCLUDER_H_INCLUDED__