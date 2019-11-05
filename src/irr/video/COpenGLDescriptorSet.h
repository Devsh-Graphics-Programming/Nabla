#ifndef __IRR_C_OPENGL_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_C_OPENGL_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/video/IGPUDescriptorSet.h"
#include "irr/macros.h"
#include "COpenGLExtensionHandler.h"
#include "COpenGLBuffer.h"
#include "COpenGLBufferView.h"
#include "COpenGLImage.h"
#include "COpenGLImageView.h"
#include "irr/video/COpenGLSampler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
namespace irr
{
namespace video
{

class COpenGLDescriptorSet : public IGPUDescriptorSet
{
	public:
		struct SMultibindParams
		{
			struct SMultibindBuffers
			{
				const GLuint* buffers = nullptr;
				const GLintptr* offsets = nullptr;
				const GLsizeiptr* sizes = nullptr;
				const uint32_t* dynOffsetIxs = nullptr;
			};
			struct SMultibindTextures
			{
				const GLuint* textures = nullptr;
				const GLuint* samplers = nullptr;
			};
			struct SMultibindTextureImages
			{
				const GLuint* textures = nullptr;
			};
        
			SMultibindBuffers ubos;
			SMultibindBuffers ssbos;
			SMultibindTextures textures;
			SMultibindTextureImages textureImages;
		};

	public:
		using IGPUDescriptorSet::IGPUDescriptorSet;

		void updateDescriptorSet(uint32_t _writeCount, const SWriteDescriptorSet* _descWrites, uint32_t _copyCount, const SCopyDescriptorSet* _descCopies) override
		{
			IGPUDescriptorSet::updateDescriptorSet(_writeCount, _descWrites, _copyCount, _descCopies);
			m_multibindParams = SMultibindParams();
			recalcMultibindParams();
		}

		const SMultibindParams& getMultibindParams() const { return m_multibindParams; }

	private:
		void recalcMultibindParams()
		{
			size_t uboCount = 0ull;//includes dynamics
			size_t ssboCount = 0ull;//includes dynamics
			size_t textureCount = 0ull;
			size_t imageCount = 0ull;
			for (const auto& desc : (*m_descriptors))
			{
				switch (desc.descriptorType)
				{
				case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
					_IRR_FALLTHROUGH;
				case asset::EDT_UNIFORM_BUFFER:
					uboCount += desc.info->size();
					break;
				case asset::EDT_STORAGE_BUFFER_DYNAMIC:
					_IRR_FALLTHROUGH;
				case asset::EDT_STORAGE_BUFFER:
					ssboCount += desc.info->size();
					break;
				case asset::EDT_UNIFORM_TEXEL_BUFFER: //GL_TEXTURE_BUFFER
					_IRR_FALLTHROUGH;
				case asset::EDT_COMBINED_IMAGE_SAMPLER:
					textureCount += desc.info->size();
					break;
				case asset::EDT_STORAGE_IMAGE:
					_IRR_FALLTHROUGH;
				case asset::EDT_STORAGE_TEXEL_BUFFER:
					imageCount += desc.info->size();
				default: break;
				}
			}

			m_names = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLuint>>(uboCount+ssboCount+(2ull*textureCount)+imageCount);
			m_offsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLintptr>>(uboCount+ssboCount);
			m_sizes = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLsizeiptr>>(uboCount+ssboCount);
			m_dynOffsetIxs = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t>>(m_offsets->size());

			const size_t uboNamesOffset = 0ull;
			const size_t ssboNamesOffset = uboCount;
			const size_t texNamesOffset = uboCount+ssboCount;
			const size_t imagNamesOffset = texNamesOffset + textureCount;
			const size_t samplerNamesOffset = imagNamesOffset + imageCount;

			const size_t uboBufOffset = 0ull;//buffer-specific offsets for `offsets` and `sizes` arrays
			const size_t ssboBufOffset = uboCount;

			auto setBufMultibindParams = [this](SMultibindParams::SMultibindBuffers& _params, size_t _namesOffset, size_t _offset) {
				_params.buffers = (*m_names).data() + _namesOffset;
				_params.offsets = (*m_offsets).data() + _offset;
				_params.sizes = (*m_sizes).data() + _offset;
				_params.dynOffsetIxs = (*m_dynOffsetIxs).data() + _offset;
			};
			setBufMultibindParams(m_multibindParams.ubos, uboNamesOffset, uboBufOffset);
			setBufMultibindParams(m_multibindParams.ssbos, ssboNamesOffset, ssboBufOffset);
			m_multibindParams.textures.textures = (*m_names).data() + texNamesOffset;
			m_multibindParams.textures.samplers = (*m_names).data() + samplerNamesOffset;
			m_multibindParams.textureImages.textures = (*m_names).data() + imagNamesOffset;

			size_t u=0ull, s=0ull, t=0ull, i=0ull;
			uint32_t dyn_offset_iter = 0u;
			const IGPUDescriptorSetLayout::SBinding* layoutBindings = m_layout->getBindings().begin();
			size_t desc_iter = 0ull;
			for (const auto& desc : (*m_descriptors))
			{
				if (desc.descriptorType==asset::EDT_UNIFORM_BUFFER || desc.descriptorType==asset::EDT_UNIFORM_BUFFER_DYNAMIC)
					for (const auto& info : (*desc.info)) {
						(*m_names)[uboNamesOffset + u] = static_cast<COpenGLBuffer*>(info.desc.get())->getOpenGLName();
						(*m_offsets)[uboBufOffset + u] = info.buffer.offset;
						(*m_sizes)[uboBufOffset + u] = info.buffer.size;
						if (desc.descriptorType == asset::EDT_UNIFORM_BUFFER_DYNAMIC)
							(*m_dynOffsetIxs)[uboBufOffset + u] = dyn_offset_iter++;
						else
							(*m_dynOffsetIxs)[uboBufOffset + u] = ~static_cast<uint32_t>(0u);

						++u;
					}
				else if (desc.descriptorType==asset::EDT_STORAGE_BUFFER || desc.descriptorType==asset::EDT_STORAGE_BUFFER_DYNAMIC)
					for (const auto& info : (*desc.info)) {
						(*m_names)[ssboNamesOffset + s] = static_cast<COpenGLBuffer*>(info.desc.get())->getOpenGLName();
						(*m_offsets)[ssboBufOffset + s] = info.buffer.offset;
						(*m_sizes)[ssboBufOffset + s] = info.buffer.size;
						if (desc.descriptorType == asset::EDT_STORAGE_BUFFER_DYNAMIC)
							(*m_dynOffsetIxs)[ssboBufOffset + s] = dyn_offset_iter++;
						else
							(*m_dynOffsetIxs)[ssboBufOffset + s] = ~static_cast<uint32_t>(0u);

						++s;
					}
				else if (desc.descriptorType==asset::EDT_COMBINED_IMAGE_SAMPLER)
				{
					size_t local_iter = 0u;
					for (const auto& info : (*desc.info)) {
						(*m_names)[texNamesOffset + t] = static_cast<COpenGLImageView*>(info.desc.get())->getOpenGLName();
						(*m_names)[samplerNamesOffset + t] =
							layoutBindings[desc_iter].samplers ? //take immutable sampler if present
							static_cast<COpenGLSampler*>(layoutBindings[desc_iter].samplers[local_iter].get())->getOpenGLName() :
							static_cast<COpenGLSampler*>(info.image.sampler.get())->getOpenGLName();
						++local_iter;
						++t;
					}
				}
				else if (desc.descriptorType==asset::EDT_UNIFORM_TEXEL_BUFFER)
					for (const auto& info : (*desc.info)) {
						(*m_names)[texNamesOffset + t] = static_cast<COpenGLBufferView*>(info.desc.get())->getOpenGLName();
						(*m_names)[samplerNamesOffset + t] = 0u;//no sampler for samplerBuffer descriptor
						++t;
					}
				else if (desc.descriptorType==asset::EDT_STORAGE_IMAGE)
					for (const auto& info : (*desc.info)) {
						(*m_names)[imagNamesOffset + i] = static_cast<COpenGLImageView*>(info.desc.get())->getOpenGLName();
						++i;
					}
				else if (desc.descriptorType==asset::EDT_STORAGE_TEXEL_BUFFER)
					for (const auto& info : (*desc.info)) {
						(*m_names)[imagNamesOffset + i] = static_cast<COpenGLBufferView*>(info.desc.get())->getOpenGLName();
						++i;
					}
				++desc_iter;
			}
		}

		SMultibindParams m_multibindParams;
		core::smart_refctd_dynamic_array<GLuint> m_names;
		core::smart_refctd_dynamic_array<GLintptr> m_offsets;
		core::smart_refctd_dynamic_array<GLsizeiptr> m_sizes;
		core::smart_refctd_dynamic_array<uint32_t> m_dynOffsetIxs;
};

}
}
#endif

#endif