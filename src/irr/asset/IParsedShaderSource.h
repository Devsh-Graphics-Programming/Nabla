#ifndef __IRR_I_PARSED_SHADER_SOURCE_H_INCLUDED__
#define __IRR_I_PARSED_SHADER_SOURCE_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ICPUBuffer.h"
#include "coreutil.h"

namespace spirv_cross
{
    class ParsedIR;
}

namespace irr { namespace asset
{

class IParsedShaderSource : public core::IReferenceCounted
{
protected:
    virtual ~IParsedShaderSource();

public:
    //! Parsing occurs instantly, during constructor execution
    IParsedShaderSource(const ICPUBuffer* _spirvBytecode);
    //! Parsing occurs upon getUnderlyingRepresentation call
    IParsedShaderSource(const ICPUBuffer* _spirvBytecode, core::defer_t);

    const spirv_cross::ParsedIR& getUnderlyingRepresentation() const
    {
        parse();
        return *m_parsed; 
    }

protected:
    void parse() const;

protected:
    mutable const spirv_cross::ParsedIR* m_parsed;
    //! Raw SPIR-V bytecode
    mutable const ICPUBuffer* m_raw;
};

}}

#endif//__IRR_I_PARSED_SHADER_SOURCE_H_INCLUDED__
