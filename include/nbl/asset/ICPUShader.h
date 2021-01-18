// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_SHADER_H_INCLUDED__
#define __NBL_ASSET_I_CPU_SHADER_H_INCLUDED__

#include <algorithm>
#include <string>


#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/IShader.h"

namespace nbl
{
namespace asset
{

//! CPU Version of Unspecialized Shader
/*
	@see IShader
	@see IAsset
*/

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

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_SHADER;
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

		const ICPUBuffer* getSPVorGLSL() const { return m_code.get(); };
		bool containsGLSL() const { return m_containsGLSL; }

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUShader*>(_other);
			if (m_containsGLSL != other->m_containsGLSL)
				return false;
			if (!m_code->canBeRestoredFrom(other->m_code.get()))
				return false;

			return true;
		}

	protected:
		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUShader*>(_other);

			if (_levelsBelow)
			{
				--_levelsBelow;

				restoreFromDummy_impl_call(m_code.get(), other->m_code.get(), _levelsBelow);
			}
		}

		//! Might be GLSL null-terminated string or SPIR-V bytecode (denoted by m_containsGLSL)
		core::smart_refctd_ptr<ICPUBuffer>	m_code;
		const bool							m_containsGLSL;
};

}
}

#endif
