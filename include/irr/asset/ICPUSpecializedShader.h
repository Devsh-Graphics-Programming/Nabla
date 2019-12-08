#ifndef __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUShader.h"
#include "irr/asset/ISpecializedShader.h"

namespace irr
{
namespace asset
{

class ICPUSpecializedShader : public IAsset, public ISpecializedShader
{
	protected:
		virtual ~ICPUSpecializedShader() = default;

	public:
		ICPUSpecializedShader(core::smart_refctd_ptr<ICPUShader>&& _unspecialized, SInfo&& _spc)
			: m_specInfo(std::move(_spc)), m_unspecialized(std::move(_unspecialized))
		{
		}


		IAsset::E_TYPE getAssetType() const override { return IAsset::ET_SPECIALIZED_SHADER; }
		size_t conservativeSizeEstimate() const override { return sizeof(void*)/*+TODO*/; }
		void convertToDummyObject() override
		{
			m_specInfo.m_entries.clear();
			m_specInfo.m_backingBuffer = nullptr;
		}

		inline E_SHADER_STAGE getStage() const { return m_specInfo.shaderStage; }

		inline void setSpecializationInfo(SInfo&& specInfo) { m_specInfo = std::move(specInfo); }
		inline const SInfo& getSpecializationInfo() const { return m_specInfo; }


		inline ICPUShader* getUnspecialized() { return m_unspecialized.get(); }
		inline const ICPUShader* getUnspecialized() const { return m_unspecialized.get(); }

	private:
		SInfo								m_specInfo;
		core::smart_refctd_ptr<ICPUShader>	m_unspecialized;
};

}}

#endif//__IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
