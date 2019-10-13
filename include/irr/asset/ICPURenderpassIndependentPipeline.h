#ifndef __IRR_I_CPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __IRR_I_CPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "irr/asset/IRenderpassIndependentPipeline.h"
#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/asset/ICPUPipelineLayout.h"

namespace irr {
namespace asset
{

class ICPURenderpassIndependentPipeline : public IRenderpassIndependentPipeline<ICPUSpecializedShader, ICPUPipelineLayout>, public IAsset
{
    using base_t = IRenderpassIndependentPipeline<ICPUSpecializedShader, ICPUPipelineLayout>;

public:
    using base_t::base_t;

    size_t conservativeSizeEstimate() const override { return sizeof(base_t); }
    void convertToDummyObject() override { }
    E_TYPE getAssetType() const override { return ET_GRAPHICS_PIPELINE; }

    inline ICPUPipelineLayout* getLayout() { return m_layout.get(); }

    inline ICPUSpecializedShader* getShaderAtStage(E_SHADER_STAGE _stage) { return m_shaders[core::findLSB<uint32_t>(_stage)].get(); }
    inline ICPUSpecializedShader* getShaderAtIndex(E_SHADER_STAGE_IX _ix) { return m_shaders[_ix].get(); }

	inline SBlendParams& getBlendParams() { return m_blendParams; }
	inline SPrimitiveAssemblyParams &getPrimitiveAssemblyParams() { return m_primAsmParams; }
	inline SRasterizationParams& getRasterizationParams() { return m_rasterParams; }

protected:
    virtual ~ICPURenderpassIndependentPipeline() = default;
};

}}

#endif