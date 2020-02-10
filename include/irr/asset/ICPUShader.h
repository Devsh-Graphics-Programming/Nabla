#ifndef __IRR_I_CPU_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SHADER_H_INCLUDED__

#include <algorithm>
#include <string>


#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/IShader.h"

namespace irr
{
namespace asset
{

class ICPUShader : public IAsset, public IShader<ICPUBuffer>
{
	protected:
		virtual ~ICPUShader() = default;

    private:
        ICPUShader(core::smart_refctd_ptr<ICPUBuffer>&& _code, bool _isGLSL) : m_code(std::move(_code)), m_containsGLSL(_isGLSL) {}

	public:
        ICPUShader(core::smart_refctd_ptr<ICPUBuffer>&& _spirv) : ICPUShader(std::move(_spirv), false) {}
		ICPUShader(const char* _glsl) : ICPUShader(core::make_smart_refctd_ptr<ICPUBuffer>(strlen(_glsl) + 1u), true)
		{
			memcpy(m_code->getPointer(), _glsl, m_code->getSize());
		}

		IAsset::E_TYPE getAssetType() const override { return IAsset::ET_SHADER; }
		size_t conservativeSizeEstimate() const override 
		{ 
			return m_code->getSize();
		}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto buf = (_depth > 0u && m_code) ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_code->clone(_depth-1u)) : m_code;
            auto cp = core::smart_refctd_ptr<ICPUShader>(new ICPUShader(std::move(buf), m_containsGLSL), core::dont_grab);
            clone_common(cp.get());

            return cp;
        }

		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
            if (isDummyObjectForCacheAliasing)
                return;
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
				m_code->convertToDummyObject(referenceLevelsBelowToConvert-1u);
		}

		const ICPUBuffer* getSPVorGLSL() const { return m_code.get(); };
		bool containsGLSL() const { return m_containsGLSL; }

	protected:
		//! Might be GLSL null-terminated string or SPIR-V bytecode (denoted by m_containsGLSL)
		core::smart_refctd_ptr<ICPUBuffer>	m_code;
		const bool							m_containsGLSL;
};

}
}

#endif//__IRR_I_CPU_SHADER_H_INCLUDED__
