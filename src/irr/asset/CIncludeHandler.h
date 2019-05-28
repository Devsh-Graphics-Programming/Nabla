#ifndef __IRR_C_INCLUDE_HANDLER_H_INCLUDED__
#define __IRR_C_INCLUDE_HANDLER_H_INCLUDED__

#include "irr/asset/IIncludeHandler.h"

#include "CFilesystemIncluder.h"
#include "CBuiltinIncluder.h"

namespace irr { namespace asset
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
    CIncludeHandler(io::IFileSystem* _filesystem)
    {
        // TODO It has to be reworked in the future
        m_includers.emplace_back(new CFilesystemIncluder(_filesystem), core::dont_grab);
        m_includers.emplace_back(new CBuiltinIncluder, core::dont_grab);
    }

    std::string getIncludeStandard(const std::string& _path) const override
    {
        return getIncluderDependentOnPath(_path)->getIncludeStandard(_path);
    }

    std::string getIncludeRelative(const std::string& _path, const std::string& _workingDirectory) const override
    {
        return getIncluderDependentOnPath(_path)->getIncludeRelative(_path, _workingDirectory);
    }

    void addBuiltinIncludeLoader(IBuiltinIncludeLoader* _inclLoader) override
    {
        static_cast<CBuiltinIncluder*>(m_includers[EII_BUILTIN].get())->addBuiltinLoader(_inclLoader);
    }

private:
    const IIncluder* getIncluderDependentOnPath(const std::string& _path) const
    {
        auto isBuiltinPath = [](const std::string& _p) {
            const char* builtinPrefixes[]{
                "irr/builtin/",
                "/irr/builtin/"
            };
            for (const char* prefix : builtinPrefixes)
                if (_p.compare(0u, strlen(prefix), prefix) == 0)
                    return true;
            return false;
        };

        return (isBuiltinPath(_path) ? m_includers[EII_BUILTIN].get() : m_includers[EII_FILESYSTEM].get());
    }
};

}}

#endif//__IRR_C_INCLUDE_HANDLER_H_INCLUDED__
