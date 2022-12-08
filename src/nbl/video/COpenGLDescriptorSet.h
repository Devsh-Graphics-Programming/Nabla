// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_DESCRIPTOR_SET_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_DESCRIPTOR_SET_H_INCLUDED__

#include "nbl/video/IGPUDescriptorSet.h"

#include "nbl/macros.h"

#include "COpenGLBuffer.h"
#include "COpenGLBufferView.h"
#include "COpenGLImageView.h"
#include "COpenGLSampler.h"

namespace nbl::video
{

class COpenGLDescriptorSet : public IGPUDescriptorSet, protected asset::impl::IEmulatedDescriptorSet<const IGPUDescriptorSetLayout>
{
	// This is OpenGL specific. The only reason it exists is because m_flatOffsets needs a coarser
	// granularity than E_DESCRIPTOR_TYPE.
	enum E_GL_DESCRIPTOR_TYPE
	{
		EGDT_UBO = 0,
		EGDT_SSBO,
		EGDT_TEXTURE,
		EGDT_IMAGE,
		EGDT_COUNT
	};

	inline E_GL_DESCRIPTOR_TYPE getGLDescriptorType(const asset::E_DESCRIPTOR_TYPE descriptorType)
	{
		switch (descriptorType)
		{
		case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
			[[fallthrough]];
		case asset::EDT_UNIFORM_BUFFER:
			return EGDT_UBO;

		case asset::EDT_STORAGE_BUFFER_DYNAMIC:
			[[fallthrough]];
		case asset::EDT_STORAGE_BUFFER:
			return EGDT_SSBO;

		case asset::EDT_UNIFORM_TEXEL_BUFFER: //GL_TEXTURE_BUFFER
			[[fallthrough]];
		case asset::EDT_COMBINED_IMAGE_SAMPLER:
			return EGDT_TEXTURE;

		case asset::EDT_STORAGE_IMAGE:
			[[fallthrough]];
		case asset::EDT_STORAGE_TEXEL_BUFFER:
			return EGDT_IMAGE;

		default:
			assert(!"Invalid code path.");
			return EGDT_COUNT;
		}
	}

	public:
		struct SMultibindParams
		{
			struct SMultibindBuffers
			{
				GLuint* buffers = nullptr;
				GLintptr* offsets = nullptr;
				GLsizeiptr* sizes = nullptr;
				uint32_t* dynOffsetIxs = nullptr;
			};
			struct SMultibindTextures
			{
				GLuint* textures = nullptr;
				GLuint* samplers = nullptr;
				GLenum* targets = nullptr;
			};
			struct SMultibindTextureImages
			{
				GLuint* textures = nullptr;
				GLenum* formats = nullptr;
			};
        
			SMultibindBuffers ubos;
			SMultibindBuffers ssbos;
			SMultibindTextures textures;
			SMultibindTextureImages textureImages;
		};

		COpenGLDescriptorSet(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout, core::smart_refctd_ptr<IDescriptorPool>&& pool)
			: IGPUDescriptorSet(std::move(dev), std::move(_layout), std::move(pool)), asset::impl::IEmulatedDescriptorSet<const IGPUDescriptorSetLayout>(m_layout.get()), m_revision(0ull)
		{
			uint32_t uboCount = 0u;//includes dynamics
			uint32_t ssboCount = 0u;//includes dynamics
			uint32_t textureCount = 0u;
			uint32_t imageCount = 0u;

			// Compute the total number of active bindings for all descriptor types.
			// uint32_t totalActiveBindingCount = 0u;
			// for (auto t = 0u; t < asset::EDT_COUNT; ++t)
			// 	totalActiveBindingCount += m_layout->m_descriptorRedirects[t].count;

			// The m_flatOffsets here currently has the problem that it will come out different based on whether there are dynamics or not.
			// m_flatOffsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t>>(totalActiveBindingCount);

#if 0
			for (uint32_t t = 0u; t < asset::EDT_COUNT; ++t)
			{
				const auto descriptorType = static_cast<asset::E_DESCRIPTOR_TYPE>(t);
				const auto& redirect = m_layout->m_descriptorRedirects[t];

				const auto activeBindingCount = redirect.count;
				for (uint32_t i = 0u; i < activeBindingCount; ++i)
				{
					const auto binding = redirect.bindings[i];

					const auto index = redirect.searchForBinding(binding);
					assert(index != redirect.Invalid);

					const auto descriptorCount = getDescriptorCount(descriptorType, binding);
					assert(descriptorCount != ~0u && "Descriptor of this type doesn't exist at this binding!");

					switch (descriptorType)
					{
					case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
						[[fallthrough]];
					case asset::EDT_UNIFORM_BUFFER:
						m_flatOffsets->operator[](index) = uboCount;
						uboCount += descriptorCount;
						break;

					case asset::EDT_STORAGE_BUFFER_DYNAMIC:
						[[fallthrough]];
					case asset::EDT_STORAGE_BUFFER:
						m_flatOffsets->operator[](index) = ssboCount;
						ssboCount += descriptorCount;
						break;

					case asset::EDT_UNIFORM_TEXEL_BUFFER: //GL_TEXTURE_BUFFER
						[[fallthrough]];
					case asset::EDT_COMBINED_IMAGE_SAMPLER:
						m_flatOffsets->operator[](index) = textureCount;
						textureCount += descriptorCount;
						break;

					case asset::EDT_STORAGE_IMAGE:
						[[fallthrough]];
					case asset::EDT_STORAGE_TEXEL_BUFFER:
						m_flatOffsets->operator[](index) = imageCount;
						imageCount += descriptorCount;
						break;

					default:
						break;
					}
				}
			}
#endif

			m_buffer2descIx = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLuint> >(uboCount+ssboCount);
			std::fill(m_buffer2descIx->begin(), m_buffer2descIx->end(), ~0u);

			m_names = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLuint> >(uboCount+ssboCount+(2ull*textureCount)+imageCount);
			std::fill(m_names->begin(),m_names->end(), 0u);
			m_offsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLintptr> >(uboCount+ssboCount);
			std::fill(m_offsets->begin(), m_offsets->end(), 0u);
			m_sizes = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLsizeiptr> >(uboCount+ssboCount);
			std::fill(m_sizes->begin(), m_sizes->end(), ~0u);
			m_dynOffsetIxs = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(m_offsets->size());
			std::fill(m_dynOffsetIxs->begin(), m_dynOffsetIxs->end(), 0u);
			m_targetsAndFormats = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLenum> >(textureCount + imageCount);
			std::fill(m_targetsAndFormats->begin(), m_targetsAndFormats->begin()+textureCount, GL_TEXTURE_2D);
			std::fill(m_targetsAndFormats->begin()+textureCount, m_targetsAndFormats->end(), GL_R8);
			
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
			m_multibindParams.textures.targets = (*m_targetsAndFormats).data();
			m_multibindParams.textureImages.textures = (*m_names).data() + imagNamesOffset;
			m_multibindParams.textureImages.formats = (*m_targetsAndFormats).data() + textureCount;

			// set up dynamic offset redirects
			m_dynamicOffsetCount = 0u;
			auto uboDescIxIter = m_buffer2descIx->begin();
			auto ssboDescIxIter = m_buffer2descIx->begin()+uboCount;

#if 0
			for (auto t = 0u; t < asset::EDT_COUNT; ++t)
			{
				const auto type = static_cast<asset::E_DESCRIPTOR_TYPE>(t);
				const auto& redirect = m_layout->m_redirects[t];
				const auto activeBindingCount = redirect.count;

				for (auto i = 0u; i < activeBindingCount; ++i)
				{
					const auto binding = redirect.bindings[i];

					const auto index = redirect.searchForBinding(binding);
					assert(index != redirect.Invalid);

					const auto offset = m_flatOffsets->operator[](index);

					const auto count = getDescriptorCount(type, binding);
					assert(count != ~0u && "Descriptor of this type doesn't exist at this binding!");

					for (uint32_t j = 0u; j < count; j++)
					{
						switch (type)
						{
						case asset::EDT_UNIFORM_BUFFER:
							*(uboDescIxIter++) = offset + j;
							m_multibindParams.ubos.dynOffsetIxs[offset + j] = ~0u;
							break;

						case asset::EDT_STORAGE_BUFFER:
							*(ssboDescIxIter++) = offset + j;
							m_multibindParams.ssbos.dynOffsetIxs[offset + j] = ~0u;
							break;

						case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
							*(uboDescIxIter++) = offset + j;
							m_multibindParams.ubos.dynOffsetIxs[offset + j] = m_dynamicOffsetCount++;
							break;

						case asset::EDT_STORAGE_BUFFER_DYNAMIC:
							*(ssboDescIxIter++) = offset + j;
							m_multibindParams.ssbos.dynOffsetIxs[offset + j] = m_dynamicOffsetCount++;
							break;

						default:
							break;
						}
					}
				}
			}
#endif
		}

		/* The following is supported:
		"If the dstBinding has fewer than descriptorCount array elements remaining starting from dstArrayElement,
		then the remainder will be used to update the subsequent binding - dstBinding+1 starting at array element zero.
		If a binding has a descriptorCount of zero, it is skipped.
		This behavior applies recursively, with the update affecting consecutive bindings as needed to update all descriptorCount descriptors.
		*/
		inline void writeDescriptorSet(const SWriteDescriptorSet& write)
		{
			assert(write.dstSet==static_cast<decltype(write.dstSet)>(this));
			assert(getDescriptorCount(write.descriptorType, write.binding)>0u);
			assert(write.count>0);

			const auto type = write.descriptorType;

			#ifdef _NBL_DEBUG
				auto layoutBinding = getLayoutBinding(write.binding);
				assert(layoutBinding);

				auto stageFlags = layoutBinding->stageFlags;
				bool usesImmutableSamplers = layoutBinding->samplers;
			#endif

			// assert(write.arrayElement+write.count<=m_descriptorInfos2[type]->size());

			auto* outDescriptorInfo = getDescriptorInfos(type, write.binding)+write.arrayElement;

			for (uint32_t i=0u; i<write.count; i++,outDescriptorInfo++)
			{
				*outDescriptorInfo = write.info[i];
				uint32_t localIx = write.arrayElement+i;
				
				// const auto index = m_layout->m_redirects[type].searchForBinding(write.binding);
				// assert(index != m_layout->m_redirects[type].Invalid && "This binding doesn't exist in the set!");

				// updateMultibindParams(type,*outDescriptorInfo,m_flatOffsets->operator[](index)+localIx,write.binding,localIx);
			}

			m_revision++;
		}
		inline void copyDescriptorSet(const SCopyDescriptorSet& copy)
		{
			assert(copy.dstSet==static_cast<decltype(copy.dstSet)>(this));
			const auto* dstGLSet = static_cast<const COpenGLDescriptorSet*>(copy.dstSet);

			assert(copy.srcSet);
			const auto* srcGLSet = static_cast<const COpenGLDescriptorSet*>(copy.srcSet);

#if 0
			asset::E_DESCRIPTOR_TYPE type = asset::EDT_COUNT;
			for (auto t = 0u; t < asset::EDT_COUNT; ++t)
			{
				// const auto& redirect = srcGLSet->m_layout->m_redirects[t];
				const auto found = redirect.searchForBinding(copy.srcBinding);
				if (found != redirect.Invalid)
				{
					type = static_cast<asset::E_DESCRIPTOR_TYPE>(t);
					break;
				}
			}
			assert(type != asset::EDT_COUNT);
#endif

			// The type of dstBinding within dstSet must be equal to the type of srcBinding within srcSet
#ifdef _NBL_DEBUG
#if 0
			asset::E_DESCRIPTOR_TYPE dstType = asset::EDT_COUNT;
			for (auto t = 0u; t < asset::EDT_COUNT; ++t)
			{
				const auto& redirect = dstGLSet->m_layout->m_redirects[t];
				const auto found = redirect.searchForBinding(copy.dstBinding);
				if (found != redirect.Invalid)
				{
					dstType = static_cast<asset::E_DESCRIPTOR_TYPE>(t);
					break;
				}
			}
			assert((dstType != asset::EDT_COUNT) && (dstType == type));
#endif
#endif
			
			// assert(copy.srcArrayElement+copy.count<=srcGLSet->m_descriptorInfos2[type]->size());
			// assert(copy.dstArrayElement+copy.count<=m_descriptorInfos2[type]->size());

#if 0
			const auto* input = srcGLSet->getDescriptorInfos(type, copy.srcBinding)+copy.srcArrayElement;
			auto* output = getDescriptorInfos(type, copy.dstBinding)+copy.dstArrayElement;

			// If srcSet is equal to dstSet, then the source and destination ranges of descriptors must not overlap
			if (this==srcGLSet)
				assert(input+copy.count<=output||output+copy.count<=input);

			for (uint32_t i=0u; i<copy.count; i++,input++,output++)
			{
				*output = *input;
				uint32_t localIx = copy.dstArrayElement+i;

				const auto index = m_layout->m_redirects[type].searchForBinding(copy.dstBinding);
				assert(index != m_layout->m_redirects[type].Invalid && "This binding doesn't exist in the set!");

				updateMultibindParams(type,*output,m_flatOffsets->operator[](index)+localIx,copy.dstBinding,localIx);
			}
#endif

			m_revision++;
		}

		inline const COpenGLBuffer* getUBO(uint32_t localIndex) const
		{
			auto ix = m_buffer2descIx->operator[](localIndex);

			// return static_cast<COpenGLBuffer*>(m_descriptorInfos->operator[](ix).desc.get());
			return nullptr;
		}

		inline const COpenGLBuffer* getSSBO(uint32_t localIndex) const
		{
			return getUBO((m_multibindParams.ssbos.offsets-m_offsets->begin())+localIndex);
		}

		inline const SMultibindParams& getMultibindParams() const { return m_multibindParams; }

		inline uint64_t getRevision() const {return m_revision;}

		inline uint32_t getDynamicOffsetCount() const {return m_dynamicOffsetCount;}

	protected:
		inline SDescriptorInfo* getDescriptorInfos(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) 
		{
			// const auto& redirect = m_layout->m_descriptorRedirects[type];
			// const auto offset = redirect[binding];
			// if (offset == redirect.Invalid)
			// 	return nullptr;
			// 
			// // return m_descriptorInfos2[type]->begin() + offset;
			// return nullptr;
			return nullptr;
		}
		inline const SDescriptorInfo* getDescriptorInfos(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) const
		{
			return getDescriptorInfos(type, binding);
		}

		inline uint32_t getDescriptorCount(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) const
		{
			constexpr auto InvalidCount = ~0u;

#if 0
			const auto& redirect = m_layout->m_redirects[type];

			const auto foundIndex = redirect.searchForBinding(binding);
			if (foundIndex == redirect.Invalid)
				return InvalidCount;

			if ((foundIndex + 1) < redirect.count)
			{
				const auto currOffset = redirect.offsets[foundIndex];
				const auto nextOffset = redirect.offsets[foundIndex + 1];
				assert(nextOffset > currOffset);
				return nextOffset - currOffset;
			}
			else
			{
				const auto currOffset = redirect.offsets[foundIndex];
				const auto totalDescriptorCount = m_layout->getTotalDescriptorCount(type);
				assert(totalDescriptorCount > currOffset);
				return totalDescriptorCount - currOffset;
			}
#endif
			return InvalidCount;
		}

		inline const video::IGPUDescriptorSetLayout::SBinding* getLayoutBinding(const uint32_t binding) const
		{
#if 0
			auto layoutBindings = m_layout->getBindings();
			auto layoutBinding = std::lower_bound(layoutBindings.begin(), layoutBindings.end(),
					video::IGPUDescriptorSetLayout::SBinding{binding,asset::EDT_COUNT,0u,asset::IShader::ESS_ALL,nullptr},
					[](const auto& a, const auto& b) -> bool {return a.binding<b.binding;});

			if ((layoutBinding == layoutBindings.end()) || (layoutBinding->binding != binding))
				return nullptr;
			else
				return layoutBinding;
#endif
			return nullptr;
		}

	private:
		inline void updateMultibindParams(asset::E_DESCRIPTOR_TYPE descriptorType, const SDescriptorInfo& info, uint32_t offset, uint32_t binding, uint32_t local_iter)
		{
			if (descriptorType==asset::EDT_UNIFORM_BUFFER || descriptorType==asset::EDT_UNIFORM_BUFFER_DYNAMIC)
			{
				m_multibindParams.ubos.buffers[offset] = static_cast<COpenGLBuffer*>(info.desc.get())->getOpenGLName();
				m_multibindParams.ubos.offsets[offset] = info.info.buffer.offset;
				m_multibindParams.ubos.sizes[offset] = info.info.buffer.size;
			}
			else if (descriptorType==asset::EDT_STORAGE_BUFFER || descriptorType==asset::EDT_STORAGE_BUFFER_DYNAMIC)
			{
				m_multibindParams.ssbos.buffers[offset] = static_cast<COpenGLBuffer*>(info.desc.get())->getOpenGLName();
				m_multibindParams.ssbos.offsets[offset] = info.info.buffer.offset;
				m_multibindParams.ssbos.sizes[offset] = info.info.buffer.size;
			}
			else if (descriptorType==asset::EDT_COMBINED_IMAGE_SAMPLER)
			{
#if 0
				auto* glimgview = static_cast<COpenGLImageView*>(info.desc.get());

				m_multibindParams.textures.textures[offset] = glimgview->getOpenGLName();

				auto layoutBindings = m_layout->getBindings();
				auto layoutBinding = std::lower_bound(layoutBindings.begin(), layoutBindings.end(),
					video::IGPUDescriptorSetLayout::SBinding{binding,asset::EDT_COUNT,0u,asset::IShader::ESS_ALL,nullptr},
					[](const auto& a, const auto& b) -> bool {return a.binding<b.binding;});
				m_multibindParams.textures.samplers[offset] = layoutBinding->samplers ? //take immutable sampler if present
						static_cast<COpenGLSampler*>(layoutBinding->samplers[local_iter].get())->getOpenGLName() :
						static_cast<COpenGLSampler*>(info.info.image.sampler.get())->getOpenGLName();
				m_multibindParams.textures.targets[offset] = glimgview->getOpenGLTarget();
#endif
			}
			else if (descriptorType==asset::EDT_UNIFORM_TEXEL_BUFFER)
			{
				m_multibindParams.textures.textures[offset] = static_cast<COpenGLBufferView*>(info.desc.get())->getOpenGLName();
				m_multibindParams.textures.samplers[offset] = 0u;//no sampler for samplerBuffer descriptor
			}
			else if (descriptorType==asset::EDT_STORAGE_IMAGE)
			{
				auto* glimgview = static_cast<COpenGLImageView*>(info.desc.get());

				m_multibindParams.textureImages.textures[offset] = glimgview->getOpenGLName();
				m_multibindParams.textureImages.formats[offset] = glimgview->getInternalFormat();
			}
			else if (descriptorType==asset::EDT_STORAGE_TEXEL_BUFFER)
			{
				auto* glbufview = static_cast<COpenGLBufferView*>(info.desc.get());

				m_multibindParams.textureImages.textures[offset] = glbufview->getOpenGLName();
				m_multibindParams.textureImages.formats[offset] = glbufview->getInternalFormat();
			}
		}

		SMultibindParams m_multibindParams;
		core::smart_refctd_dynamic_array<uint32_t> m_flatOffsets;

		core::smart_refctd_dynamic_array<uint32_t> m_buffer2descIx;

		core::smart_refctd_dynamic_array<GLuint> m_names;
		core::smart_refctd_dynamic_array<GLintptr> m_offsets;
		core::smart_refctd_dynamic_array<GLsizeiptr> m_sizes;
		core::smart_refctd_dynamic_array<GLenum> m_targetsAndFormats;
		core::smart_refctd_dynamic_array<uint32_t> m_dynOffsetIxs;

		uint64_t m_revision;
		uint32_t m_dynamicOffsetCount;
};

}

#endif