#ifndef __IRR_I_INCLUDER_H_INCLUDED__
#define __IRR_I_INCLUDER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/core/Types.h"
#include "IFileSystem.h"
#include <string>

namespace irr { namespace asset
{

class IIncluder : public core::IReferenceCounted
{
protected:
    core::vector<std::string> m_searchDirectories;

    virtual ~IIncluder() = default;

public:
    IIncluder() : m_searchDirectories{""} {}

    virtual void addSearchDirectory(const std::string& _searchDir) { m_searchDirectories.push_back(_searchDir); }

    std::string getInclude(const std::string& _path) const
    {
        for (const std::string& searchDir : m_searchDirectories)
        {
            io::path path = searchDir.c_str();
            path += _path.c_str();
            io::IFileSystem::flattenFilename(path);
            path[path.size()-1] = 0; // for some reason flattenFilename() adds to the end
            std::string res = getInclude_internal(path.c_str());
            if (!res.empty())
                return res;
        }
        return {};
    }

protected:
    //! Always gets absolute path
    virtual std::string getInclude_internal(const std::string& _path) const = 0;
};

}}

#endif//__IRR_I_INCLUDER_H_INCLUDED__