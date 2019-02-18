#ifndef __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUShader.h"
#include "irr/asset/ShaderCommons.h"

namespace irr { namespace asset
{

class ICPUSpecializedShader : public IAsset
{
protected:
    virtual ~ICPUSpecializedShader()
    {
        if (m_specInfo)
            m_specInfo->drop();
        if (m_unspecialized)
            m_unspecialized->drop();
    }

public:
    ICPUSpecializedShader(const ICPUShader* _unspecialized, const ISpecializationInfo* _spc)
        : m_unspecialized{_unspecialized}, m_specInfo{_spc}
    {
        m_unspecialized->grab();
        m_specInfo->grab();
    }

    IAsset::E_TYPE getAssetType() const override { return IAsset::ET_SPECIALIZED_SHADER; }
    size_t conservativeSizeEstimate() const override { return 0u; /* TODO: ???? */ }
    void convertToDummyObject() override { }

    inline const ISpecializationInfo* getSpecializationInfo() const { return m_specInfo; }
    inline const ICPUShader* getUnspecialized() const { return m_unspecialized; }

private:
    const ICPUShader* m_unspecialized;
    const ISpecializationInfo* m_specInfo;
};

}}

#endif//__IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
