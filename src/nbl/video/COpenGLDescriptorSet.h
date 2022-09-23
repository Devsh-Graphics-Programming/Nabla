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

		COpenGLDescriptorSet(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout) : IGPUDescriptorSet(std::move(dev), std::move(_layout)), asset::impl::IEmulatedDescriptorSet<const IGPUDescriptorSetLayout>(m_layout.get()), m_revision(0ull)
		{
			m_flatOffsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t>>(m_bindingInfo->size());
			uint32_t uboCount = 0u;//includes dynamics
			uint32_t ssboCount = 0u;//includes dynamics
			uint32_t textureCount = 0u;
			uint32_t imageCount = 0u;
			for (uint32_t i=0u; i<m_bindingInfo->size(); i++)
			{
				auto count = getDescriptorCountAtIndex(i);
				switch (m_bindingInfo->operator[](i).descriptorType)
				{
					case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
						[[fallthrough]];
					case asset::EDT_UNIFORM_BUFFER:
						m_flatOffsets->operator[](i) = uboCount;
						uboCount += count;
						break;
					case asset::EDT_STORAGE_BUFFER_DYNAMIC:
						[[fallthrough]];
					case asset::EDT_STORAGE_BUFFER:
						m_flatOffsets->operator[](i) = ssboCount;
						ssboCount += count;
						break;
					case asset::EDT_UNIFORM_TEXEL_BUFFER: //GL_TEXTURE_BUFFER
						[[fallthrough]];
					case asset::EDT_COMBINED_IMAGE_SAMPLER:
						m_flatOffsets->operator[](i) = textureCount;
						textureCount += count;
						break;
					case asset::EDT_STORAGE_IMAGE:
						[[fallthrough]];
					case asset::EDT_STORAGE_TEXEL_BUFFER:
						m_flatOffsets->operator[](i) = imageCount;
						imageCount += count;
						break;
					default:
						break;
				}
			}
			
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
			for (size_t i=0u; i<m_bindingInfo->size(); i++)
			{
				const auto& info = m_bindingInfo->operator[](i);
				auto offset = m_flatOffsets->operator[](i);
				for (uint32_t j=0u; j<getDescriptorCountAtIndex(i); j++)
				switch (info.descriptorType)
				{
					case asset::EDT_UNIFORM_BUFFER:
						*(uboDescIxIter++) = offset+j;
						m_multibindParams.ubos.dynOffsetIxs[offset+j] = ~static_cast<uint32_t>(0u);
						break;
					case asset::EDT_STORAGE_BUFFER:
						*(ssboDescIxIter++) = offset+j;
						m_multibindParams.ssbos.dynOffsetIxs[offset+j] = ~static_cast<uint32_t>(0u);
						break;
					case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
						*(uboDescIxIter++) = offset+j;
						m_multibindParams.ubos.dynOffsetIxs[offset+j] = m_dynamicOffsetCount++;
						break;
					case asset::EDT_STORAGE_BUFFER_DYNAMIC:
						*(ssboDescIxIter++) = offset+j;
						m_multibindParams.ssbos.dynOffsetIxs[offset+j] = m_dynamicOffsetCount++;
						break;
					default:
						break;
				}
			}
		}

		/* The following is supported:
		"If the dstBinding has fewer than descriptorCount array elements remaining starting from dstArrayElement,
		then the remainder will be used to update the subsequent binding - dstBinding+1 starting at array element zero.
		If a binding has a descriptorCount of zero, it is skipped.
		This behavior applies recursively, with the update affecting consecutive bindings as needed to update all descriptorCount descriptors.
		*/
		inline void writeDescriptorSet(const SWriteDescriptorSet& _write)
		{
			assert(_write.dstSet==static_cast<decltype(_write.dstSet)>(this));
			assert(m_bindingInfo);

			assert(_write.binding<m_bindingInfo->size());
			assert(getDescriptorCountAtIndex(_write.binding)>0u);
			assert(_write.count>0);
			const auto type = _write.descriptorType;
			#ifdef _NBL_DEBUG
				auto info = m_bindingInfo->operator[](_write.binding);
				assert(type==info.descriptorType);
				auto layoutBinding = getLayoutBinding(_write.binding);
				auto stageFlags = layoutBinding->stageFlags;
				bool usesImmutableSamplers = layoutBinding->samplers;
			#endif
			assert(_write.arrayElement+_write.count<=m_descriptors->size());
			auto* output = getDescriptors(_write.binding)+_write.arrayElement;
			for (uint32_t i=0u; i<_write.count; i++,output++)
			{
				#ifdef _NBL_DEBUG
					auto found = getBindingInfo(output-m_descriptors->begin());
					assert(found->descriptorType == type);
					layoutBinding = getLayoutBinding(found-m_bindingInfo->begin());
					assert(layoutBinding->stageFlags==stageFlags);
					assert((!!layoutBinding->samplers)==usesImmutableSamplers);
				#endif
				*output = _write.info[i];
				uint32_t localIx = _write.arrayElement+i;
				updateMultibindParams(type,*output,m_flatOffsets->operator[](_write.binding)+localIx,_write.binding,localIx);
			}

			m_revision++;
		}
		inline void copyDescriptorSet(const SCopyDescriptorSet& _copy)
		{
			assert(_copy.dstSet==static_cast<decltype(_copy.dstSet)>(this));
			assert(_copy.srcSet);
			const auto* srcGLSet = static_cast<const COpenGLDescriptorSet*>(_copy.srcSet);
			assert(m_bindingInfo && srcGLSet->m_bindingInfo);

			assert(_copy.srcBinding<srcGLSet->m_bindingInfo->size());
			assert(_copy.dstBinding<m_bindingInfo->size());
			assert(_copy.srcArrayElement+_copy.count<=srcGLSet->m_descriptors->size());
			assert(_copy.dstArrayElement+_copy.count<=m_descriptors->size());
			// The type of dstBinding within dstSet must be equal to the type of srcBinding within srcSet
			const auto type = srcGLSet->m_bindingInfo->operator[](_copy.srcBinding).descriptorType;
			assert(type==m_bindingInfo->operator[](_copy.dstBinding).descriptorType);
			
			const auto* input = srcGLSet->getDescriptors(_copy.srcBinding)+_copy.srcArrayElement;
			auto* output = getDescriptors(_copy.dstBinding)+_copy.dstArrayElement;
			// If srcSet is equal to dstSet, then the source and destination ranges of descriptors must not overlap
			if (this==srcGLSet)
				assert(input+_copy.count<=output||output+_copy.count<=input);
			for (uint32_t i=0u; i<_copy.count; i++,input++,output++)
			{
				#ifdef _NBL_DEBUG
					auto foundIn = getBindingInfo(input-srcGLSet->m_descriptors->begin());
					auto foundOut = getBindingInfo(output-m_descriptors->begin());
					assert(foundIn->descriptorType==foundOut->descriptorType);

					// TODO: fix this debug code
					//auto inLayoutBinding = getLayoutBinding(foundIn-srcGLSet->m_bindingInfo->begin());
					//auto outLayoutBinding = getLayoutBinding(foundOut-m_bindingInfo->begin());
					//assert(outLayoutBinding->stageFlags==inLayoutBinding->stageFlags);
					//assert((!outLayoutBinding->samplers)==(!inLayoutBinding->samplers));
				#endif
				*output = *input;
				uint32_t localIx = _copy.dstArrayElement+i;
				updateMultibindParams(type,*output,m_flatOffsets->operator[](_copy.dstBinding)+localIx,_copy.dstBinding,localIx);
			}

			m_revision++;
		}

		inline const COpenGLBuffer* getUBO(uint32_t localIndex) const
		{
			auto ix = m_buffer2descIx->operator[](localIndex);
			return static_cast<COpenGLBuffer*>(m_descriptors->operator[](ix).desc.get());
		}

		inline const COpenGLBuffer* getSSBO(uint32_t localIndex) const
		{
			return getUBO((m_multibindParams.ssbos.offsets-m_offsets->begin())+localIndex);
		}

		inline const SMultibindParams& getMultibindParams() const { return m_multibindParams; }

		inline uint64_t getRevision() const {return m_revision;}

		inline uint32_t getDynamicOffsetCount() const {return m_dynamicOffsetCount;}

	protected:
		inline SDescriptorInfo* getDescriptors(uint32_t index) 
		{ 
			const auto& info = m_bindingInfo->operator[](index);
			return m_descriptors->begin()+info.offset;
		}
		inline const SDescriptorInfo* getDescriptors(uint32_t index) const
		{
			const auto& info = m_bindingInfo->operator[](index);
			return m_descriptors->begin() + info.offset;
		}

		inline uint32_t getDescriptorCountAtIndex(uint32_t index) const
		{
			const auto& info = m_bindingInfo->operator[](index);
			if (index + 1u != m_bindingInfo->size())
			{
				const auto& info1 = m_bindingInfo->operator[](index + 1u);
				return info1.offset - info.offset;
			}
			else
				return m_descriptors->size() - info.offset;
		}

		inline const SBindingInfo* getBindingInfo(uint32_t offset) const
		{
			auto found = std::upper_bound(	m_bindingInfo->begin(),m_bindingInfo->end(),SBindingInfo{offset,asset::EDT_COUNT},
											[](const auto& a, const auto& b) -> bool {return a.offset<b.offset;});
			assert(found!=m_bindingInfo->begin());
			return found-1;
		}

		inline const video::IGPUDescriptorSetLayout::SBinding* getLayoutBinding(uint32_t binding) const
		{
			auto layoutBindings = m_layout->getBindings();
			auto layoutBinding = std::lower_bound(layoutBindings.begin(), layoutBindings.end(),
					video::IGPUDescriptorSetLayout::SBinding{binding,asset::EDT_COUNT,0u,asset::IShader::ESS_ALL,nullptr},
					[](const auto& a, const auto& b) -> bool {return a.binding<b.binding;});
			assert(layoutBinding!=layoutBindings.end());
			return layoutBinding;
		}

	private:
		inline void updateMultibindParams(asset::E_DESCRIPTOR_TYPE descriptorType, const SDescriptorInfo& info, uint32_t offset, uint32_t binding, uint32_t local_iter)
		{
			if (descriptorType==asset::EDT_UNIFORM_BUFFER || descriptorType==asset::EDT_UNIFORM_BUFFER_DYNAMIC)
			{
				m_multibindParams.ubos.buffers[offset] = static_cast<COpenGLBuffer*>(info.desc.get())->getOpenGLName();
				m_multibindParams.ubos.offsets[offset] = info.buffer.offset;
				m_multibindParams.ubos.sizes[offset] = info.buffer.size;
			}
			else if (descriptorType==asset::EDT_STORAGE_BUFFER || descriptorType==asset::EDT_STORAGE_BUFFER_DYNAMIC)
			{
				m_multibindParams.ssbos.buffers[offset] = static_cast<COpenGLBuffer*>(info.desc.get())->getOpenGLName();
				m_multibindParams.ssbos.offsets[offset] = info.buffer.offset;
				m_multibindParams.ssbos.sizes[offset] = info.buffer.size;
			}
			else if (descriptorType==asset::EDT_COMBINED_IMAGE_SAMPLER)
			{
				auto* glimgview = static_cast<COpenGLImageView*>(info.desc.get());

				m_multibindParams.textures.textures[offset] = glimgview->getOpenGLName();

				auto layoutBindings = m_layout->getBindings();
				auto layoutBinding = std::lower_bound(layoutBindings.begin(), layoutBindings.end(),
					video::IGPUDescriptorSetLayout::SBinding{binding,asset::EDT_COUNT,0u,asset::IShader::ESS_ALL,nullptr},
					[](const auto& a, const auto& b) -> bool {return a.binding<b.binding;});
				m_multibindParams.textures.samplers[offset] = layoutBinding->samplers ? //take immutable sampler if present
						static_cast<COpenGLSampler*>(layoutBinding->samplers[local_iter].get())->getOpenGLName() :
						static_cast<COpenGLSampler*>(info.image.sampler.get())->getOpenGLName();
				m_multibindParams.textures.targets[offset] = glimgview->getOpenGLTarget();
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