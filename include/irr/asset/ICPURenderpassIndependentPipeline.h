#ifndef __IRR_I_CPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __IRR_I_CPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "irr/asset/IRenderpassIndependentPipeline.h"
#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/asset/ICPUPipelineLayout.h"
#include "irr/asset/IPipelineMetadata.h"

namespace irr
{
namespace asset
{

class ICPURenderpassIndependentPipeline : public IRenderpassIndependentPipeline<ICPUSpecializedShader, ICPUPipelineLayout>, public IAsset
{
		using base_t = IRenderpassIndependentPipeline<ICPUSpecializedShader, ICPUPipelineLayout>;

	public:
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t DESC_SET_HIERARCHYLEVELS_BELOW = 0u;
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t IMAGEVIEW_HIERARCHYLEVELS_BELOW = 1u;
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t IMAGE_HIERARCHYLEVELS_BELOW = 2u;


		using base_t::base_t;

		size_t conservativeSizeEstimate() const override { return sizeof(base_t); }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			if (referenceLevelsBelowToConvert)
			{
                //intentionally parent is not converted
                --referenceLevelsBelowToConvert;
				m_layout->convertToDummyObject(referenceLevelsBelowToConvert);
				for (auto i=0u; i<SHADER_STAGE_COUNT; i++)
                    if (m_shaders[i])
					    m_shaders[i]->convertToDummyObject(referenceLevelsBelowToConvert);
			}
		}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            core::smart_refctd_ptr<ICPUPipelineLayout> layout = (_depth > 0u && m_layout) ? core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(m_layout->clone(_depth-1u)) : m_layout;

            std::array<core::smart_refctd_ptr<ICPUSpecializedShader>, SHADER_STAGE_COUNT> shaders;
            for (uint32_t i = 0u; i < shaders.size(); ++i)
                (_depth > 0u && m_shaders[i]) ? m_shaders[i]->clone(_depth-1u) : m_shaders[i];
            std::array<ICPUSpecializedShader*, SHADER_STAGE_COUNT> shaders_raw;
            for (uint32_t i = 0u; i < shaders.size(); ++i)
                shaders_raw[i] = shaders[i].get();
            std::sort(shaders_raw.begin(), shaders_raw.end(), [](ICPUSpecializedShader* a, ICPUSpecializedShader* b) { return (a && !b); });

            auto cp = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(layout),
                shaders_raw.data(), &*std::find(shaders_raw.begin(), shaders_raw.end(), nullptr),
                m_vertexInputParams, m_blendParams, m_primAsmParams, m_rasterParams
            );
            clone_common(cp.get());

            return cp;
        }

		E_TYPE getAssetType() const override { return ET_RENDERPASS_INDEPENDENT_PIPELINE; }

		inline ICPUPipelineLayout* getLayout() { return m_layout.get(); }
		const inline ICPUPipelineLayout* getLayout() const { return m_layout.get(); }

		inline ICPUSpecializedShader* getShaderAtStage(ISpecializedShader::E_SHADER_STAGE _stage) { return m_shaders[core::findLSB<uint32_t>(_stage)].get(); }
		inline ICPUSpecializedShader* getShaderAtIndex(uint32_t _ix) { return m_shaders[_ix].get(); }
		inline const ICPUSpecializedShader* getShaderAtIndex(uint32_t _ix) const { return m_shaders[_ix].get(); }

		inline SBlendParams& getBlendParams() { return m_blendParams; }
		inline const SBlendParams& getBlendParams() const { return m_blendParams; }
		inline SPrimitiveAssemblyParams& getPrimitiveAssemblyParams() { return m_primAsmParams; }
		inline const SPrimitiveAssemblyParams& getPrimitiveAssemblyParams() const { return m_primAsmParams; }
		inline SRasterizationParams& getRasterizationParams() { return m_rasterParams; }
		inline const SRasterizationParams& getRasterizationParams() const { return m_rasterParams; }
		inline SVertexInputParams& getVertexInputParams() { return m_vertexInputParams; }
		inline const SVertexInputParams& getVertexInputParams() const { return m_vertexInputParams; }

		inline void setShaderAtStage(ISpecializedShader::E_SHADER_STAGE _stage, ICPUSpecializedShader* _shdr) { m_shaders[core::findLSB<uint32_t>(_stage)] = core::smart_refctd_ptr<ICPUSpecializedShader>(_shdr); }
		inline void setShaderAtIndex(uint32_t _ix, ICPUSpecializedShader* _shdr) { m_shaders[_ix] = core::smart_refctd_ptr<ICPUSpecializedShader>(_shdr); }

	protected:
		virtual ~ICPURenderpassIndependentPipeline() = default;
};

}
}

#endif