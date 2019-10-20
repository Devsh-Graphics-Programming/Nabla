#ifndef __IRR_I_CPU_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SHADER_H_INCLUDED__

#include <algorithm>
#include <string>
#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/ShaderCommons.h"
#include "irr/core/SRange.h"

namespace spirv_cross
{
    class ParsedIR;
    class Compiler;
    struct SPIRType;
}
namespace irr {
namespace asset
{

class ICPUShader : public IAsset
{
protected:
    virtual ~ICPUShader()
    {
        if (m_code)
            m_code->drop();
    }

public:
    ICPUShader(ICPUBuffer* _spirv);
    ICPUShader(const char* _glsl);

    IAsset::E_TYPE getAssetType() const override { return IAsset::ET_SHADER; }
    size_t conservativeSizeEstimate() const override 
    { 
        return m_code->getSize();
    }
    void convertToDummyObject() override { }

    const ICPUBuffer* getSPVorGLSL() const { return m_code; };
    bool containsGLSL() const { return m_containsGLSL; }

    static void insertGLSLExtensionsDefines(std::string& _glsl, const core::SRange<const std::string>& _exts);

protected:
    //! Might be GLSL null-terminated string or SPIR-V bytecode (denoted by m_containsGLSL)
    ICPUBuffer* m_code;
    const bool m_containsGLSL;
};

}}

#endif//__IRR_I_CPU_SHADER_H_INCLUDED__
