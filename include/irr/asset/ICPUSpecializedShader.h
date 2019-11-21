#ifndef __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUShader.h"
#include "irr/asset/ShaderCommons.h"

namespace irr
{
namespace asset
{

class ICPUSpecializedShader : public IAsset
{
	protected:
		virtual ~ICPUSpecializedShader() = default;

	public:
		ICPUSpecializedShader(core::smart_refctd_ptr<ICPUShader>&& _unspecialized, core::smart_refctd_ptr<const ISpecializationInfo>&& _spc)
			: m_unspecialized(std::move(_unspecialized)), m_specInfo(std::move(_spc))
		{
		}

		IAsset::E_TYPE getAssetType() const override { return IAsset::ET_SPECIALIZED_SHADER; }
		size_t conservativeSizeEstimate() const override { return 0u; /* TODO: ???? */ }
		void convertToDummyObject() override { }

		inline E_SHADER_STAGE getStage() const { return m_specInfo->shaderStage; }
		inline const ISpecializationInfo* getSpecializationInfo() const { return m_specInfo.get(); }
		inline const ICPUShader* getUnspecialized() const { return m_unspecialized.get(); }
		inline ICPUShader* getUnspecialized() { return m_unspecialized.get(); }

	private:
		core::smart_refctd_ptr<ICPUShader>					m_unspecialized;
		core::smart_refctd_ptr<const ISpecializationInfo>	m_specInfo;
};

}}

#endif//__IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
