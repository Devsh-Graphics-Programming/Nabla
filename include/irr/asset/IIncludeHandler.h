#ifndef __IRR_I_INCLUDE_HANDLER_H_INCLUDED__
#define __IRR_I_INCLUDE_HANDLER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr { namespace asset
{

class IIncludeHandler : public core::IReferenceCounted
{
protected:
    virtual ~IIncludeHandler() = default;

public:
    virtual std::string getIncludeStandard(const std::string& _path) const = 0;
    virtual std::string getIncludeRelative(const std::string& _path, const std::string& _workingDirectory) const = 0;

    virtual void addBuiltinIncludeLoader(IBuiltinIncludeLoader* _inclLoader) = 0;
};

}}

#endif//__IRR_I_INCLUDE_HANDLER_H_INCLUDED__
