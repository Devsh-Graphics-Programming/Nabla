#ifndef __IRR_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"



namespace irr { namespace asset
{

class IBuiltinIncludeLoader : public core::IReferenceCounted
{
public:
    virtual ~IBuiltinIncludeLoader() = default;

    virtual std::string getBuiltinInclude(const std::string& _name) const = 0;

    //! @returns Path relative to /irr/builtin/
    virtual const char* getVirtualDirectoryName() const = 0;
};

}}

#endif//__IRR_I_BUILTIN_INCLUDE_LOADER_H_INCLUDED__