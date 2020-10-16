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

class ICPUShader : public IAsset, public IShader
{
	protected:
		virtual ~ICPUShader() = default;

    private:
        ICPUShader(core::smart_refctd_ptr<ICPUBuffer>&& _code, bool _isGLSL) : m_code(std::move(_code)), m_containsGLSL(_isGLSL) {}

	public:
        ICPUShader(core::smart_refctd_ptr<ICPUBuffer>&& _spirv) : ICPUShader(std::move(_spirv), false) {}
        ICPUShader(core::smart_refctd_ptr<ICPUBuffer>&& _glsl, buffer_contains_glsl_t _buffer_contains_glsl) : ICPUShader(std::move(_glsl), true) {}
		ICPUShader(const char* _glsl) : ICPUShader(core::make_smart_refctd_ptr<ICPUBuffer>(strlen(_glsl) + 1u), true)
		{
			memcpy(m_code->getPointer(), _glsl, m_code->getSize());
		}

		_IRR_STATIC_INLINE_CONSTEXPR auto AssetType = ET_SHADER;
		inline E_TYPE getAssetType() const override { return AssetType; }

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
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
				m_code->convertToDummyObject(referenceLevelsBelowToConvert-1u);
		}

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			if (!IAsset::canBeRestoredFrom(_other))
				return false;

			auto* other = static_cast<const ICPUShader*>(_other);
			if (m_containsGLSL != other->m_containsGLSL)
				return false;

			return true;
		}

		const ICPUBuffer* getSPVorGLSL() const { return m_code.get(); };
		bool containsGLSL() const { return m_containsGLSL; }

	private:
		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUShader*>(_other);

			if (_levelsBelow)
			{
				--_levelsBelow;

				m_code->restoreFromDummy(other->m_code.get(), _levelsBelow);
			}
		}

	protected:
		//! Might be GLSL null-terminated string or SPIR-V bytecode (denoted by m_containsGLSL)
		core::smart_refctd_ptr<ICPUBuffer>	m_code;
		const bool							m_containsGLSL;
};

}
}

#endif//__IRR_I_CPU_SHADER_H_INCLUDED__
