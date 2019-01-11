#ifndef __IRR_I_CPU_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SHADER_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ISPIR_VProgram.h"

namespace irr { namespace asset
{

class ICPUShader : public IAsset
{
protected:
    virtual ~ICPUShader()
    {
        if (m_spirvProgram)
            m_spirvProgram->drop();
    }

public:
    explicit ICPUShader(ISPIR_VProgram* _spirvProgram) : m_spirvProgram{_spirvProgram}, m_containsSpirv{_spirvProgram!=nullptr}
    {
        if (m_spirvProgram)
            m_spirvProgram->grab();
    }
    explicit ICPUShader(const std::string& _vkglsl) : m_spirvProgram{nullptr}, m_containsSpirv{false}
    {
        ICPUBuffer* vkglsl = new ICPUBuffer(_vkglsl.size()+1);
        memcpy(vkglsl->getPointer(), _vkglsl.data(), vkglsl->getSize());
        reinterpret_cast<char*>(vkglsl->getPointer())[_vkglsl.size()] = 0;

        m_spirvProgram = new ISPIR_VProgram(vkglsl);
        vkglsl->drop();
    }

    const ISPIR_VProgram* getSPIR_VProgram()
    {
        if (m_containsSpirv)
        {
            /* use GLSL code held in current m_spirvProgram,
            take care of #include directives and insert extensions #define's,
            compile it to SPIR-V (IGLSLCompiler),
            create new ISPIR_VProgram (while dropping current one)
            */
            m_containsSpirv = false;
        }
        return m_spirvProgram;
    }

protected:
    bool m_containsSpirv;
    ISPIR_VProgram* m_spirvProgram;
};

}}

#endif//__IRR_I_CPU_SHADER_H_INCLUDED__
