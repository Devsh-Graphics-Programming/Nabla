// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __NBL_ASSET_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/core/SRange.h"
#include "nbl/asset/ISpecializedShader.h"

namespace nbl
{
namespace asset
{

// TODO: move this to appropriate class
enum E_DESCRIPTOR_TYPE : uint8_t
{
    EDT_COMBINED_IMAGE_SAMPLER = 0,
    EDT_STORAGE_IMAGE,
    EDT_UNIFORM_TEXEL_BUFFER,
    EDT_STORAGE_TEXEL_BUFFER,
    EDT_UNIFORM_BUFFER,
    EDT_STORAGE_BUFFER,
    EDT_UNIFORM_BUFFER_DYNAMIC,
    EDT_STORAGE_BUFFER_DYNAMIC,
    EDT_INPUT_ATTACHMENT,
	// Provided by VK_KHR_acceleration_structure
	EDT_ACCELERATION_STRUCTURE,

	// Support for the following is not available:
	// Provided by VK_EXT_inline_uniform_block
	// EDT_INLINE_UNIFORM_BLOCK_EXT,
	// Provided by VK_NV_ray_tracing
	// EDT_ACCELERATION_STRUCTURE_NV = 1000165000,
	// Provided by VK_VALVE_mutable_descriptor_type
	// EDT_MUTABLE_VALVE = 1000351000,

	EDT_COUNT
};

//! Interface class for Descriptor Set Layouts
/*
	The descriptor set layout specifies the bindings (in the shader GLSL
	interfaces), counts and types of resources like:

	- UBO
	- SSBO
	- combined image samplers
	- storage images

	that will be used by the shader stages (the shader stage flag, vertex, fragment, etc.).

	a GLSL shader declares resources with:

	\code{.glsl}
	layout(set = N, binding = M) TYPE name[K];
	\code

	The following example shows how to set up one SBinding to create
	a basic DescriptorSetLayout with above formula:

	\code{.cpp}
	// We will use set N, binding M and count K and descriptor type X

	asset::ICPUDescriptorSetLayout::SBinding binding;
	binding.count = K;
	binding.binding = M;
	binding.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
	binding.type = X; // It might be an asset::EDT_UNIFORM_BUFFER for instance
	auto descriptorSetLayout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(&binding, &binding + 1);

	// Assuming that set N = 1, you execute std::move() one second constructor's field of available descriptor set layouts
	auto pipelineLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(nullptr, nullptr, nullptr, std::move(descriptorSetLayout), nullptr, nullptr);
	\code

	@see IReferenceCounted
*/

template<typename SamplerType>
class NBL_API IDescriptorSetLayout : public virtual core::IReferenceCounted
{
	struct SBindingRedirect
	{
		static constexpr inline uint32_t Invalid = ~0u;

		SBindingRedirect() : bindings(nullptr), offsets(nullptr), count(0ull) {}

		SBindingRedirect(const size_t _count) : count(_count)
		{
			bindings = std::make_unique<uint32_t[]>(count << 1);
			offsets = bindings.get() + count;
		}

		// For a given binding number `binding` in `bindings`, return its offset in `offsets`.
		inline uint32_t operator[](const uint32_t binding) const
		{
			const uint32_t index = searchForBinding(binding);
			if (index == Invalid)
				return Invalid;

			return offsets[index];
		}

		// Returns index into the `bindings` and `offsets` array for the given binding number `binding.
		// Assumes `bindings` is sorted.
		inline uint32_t searchForBinding(const uint32_t binding) const
		{
			if (!bindings || !offsets || (count == 0ull))
				return Invalid;

			auto found = std::lower_bound(bindings.get(), bindings.get() + count, binding);

			if ((found >= bindings.get() + count) || (*found != binding))
				return Invalid;

			const uint32_t foundIndex = found - bindings.get();
			assert(foundIndex < count);
			return foundIndex;
		}

		std::unique_ptr<uint32_t[]> bindings;
		uint32_t* offsets;
		size_t count;
	};

	public:
		using sampler_type = SamplerType;

		struct SBinding
		{
			enum class E_CREATE_FLAGS : uint8_t
			{
				ECF_NONE							= 0,
				ECF_UPDATE_AFTER_BIND_BIT			= 1u << 1,
				ECF_UPDATE_UNUSED_WHILE_PENDING_BIT = 1u << 2,
				ECF_PARTIALLY_BOUND_BIT				= 1u << 3
			};

			uint32_t binding;
			E_DESCRIPTOR_TYPE type;
			core::bitflag<E_CREATE_FLAGS> createFlags;
			IShader::E_SHADER_STAGE stageFlags; // TODO(achal): Should make a core::bitflag out of this as well?
			uint32_t count;
			// Use this if you want an immutable sampler that is baked into the DS layout itself.
			// If its `nullptr` then the sampler used is mutable and can be specified while writing the image descriptor to a binding while updating the DS.
			const core::smart_refctd_ptr<sampler_type>* samplers;

			bool operator<(const SBinding& rhs) const
			{
				if (binding==rhs.binding)
				{
					// should really assert here
					if (type==rhs.type)
					{
						if (count==rhs.count)
						{
							if (stageFlags==rhs.stageFlags)
							{
								if (createFlags.value == rhs.createFlags.value)
								{
									for (uint32_t i = 0u; i < count; i++)
									{
										if (samplers[i] == rhs.samplers[i])
											continue;
										return samplers[i] < rhs.samplers[i];
									}
									return false;
								}
								return createFlags.value < rhs.createFlags.value;
							}
							return stageFlags<rhs.stageFlags;
						}
						return count<rhs.count;
					}
					return type<rhs.type;
				}
				return binding<rhs.binding;
			}
			bool operator==(const SBinding& rhs) const
			{
				if (binding != rhs.binding)
					return false;
				if (type != rhs.type)
					return false;
				if (count != rhs.count)
					return false;
				if (stageFlags != rhs.stageFlags)
					return false;
				if (createFlags.value != rhs.createFlags.value)
					return false;

				if (!samplers && !rhs.samplers)
					return true;
				else if ((samplers && !rhs.samplers) || (!samplers && rhs.samplers))
					return false;

				for (uint32_t i = 0u; i < count; ++i)
					if (samplers[i] != rhs.samplers[i])
						return false;

				return true;
			}
			bool operator!=(const SBinding& rhs) const
			{
				const bool equal = operator==(rhs);
				return !equal;
			}
		};
		
		// utility functions
		static inline void fillBindingsSameType(SBinding* bindings, uint32_t count, E_DESCRIPTOR_TYPE type, const uint32_t* counts=nullptr, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			for (auto i=0u; i<count; i++)
			{
				bindings[i].binding = i;
				bindings[i].type = type;
				bindings[i].count = counts ? counts[i]:1u;
				bindings[i].stageFlags = stageAccessFlags ? stageAccessFlags[i]:asset::IShader::ESS_ALL;
				bindings[i].samplers = nullptr;
			}
		}

		//
		IDescriptorSetLayout(const SBinding* const _begin, const SBinding* const _end) : 
			m_bindings((_end-_begin) ? core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SBinding>>(_end-_begin) : nullptr)
		{
			struct SBindingRedirectBuildInfo
			{
				uint32_t binding;
				uint32_t count;

				inline bool operator< (const SBindingRedirectBuildInfo& other) const { return binding < other.binding; }
			};

			core::vector<SBindingRedirectBuildInfo> buildInfo_descriptors[asset::EDT_COUNT];
			core::vector<SBindingRedirectBuildInfo> buildInfo_samplers;

			for (auto b = _begin; b != _end; ++b)
			{
				buildInfo_descriptors[b->type].emplace_back(b->binding, b->count);
				if (b->type == EDT_COMBINED_IMAGE_SAMPLER && b->samplers == nullptr)
				{
					buildInfo_samplers.emplace_back(b->binding, b->count);
					m_mutableSamplerCount += b->count;
				}
			}

			for (auto type = 0u; type < asset::EDT_COUNT; ++type)
				m_descriptorRedirects[type] = SBindingRedirect(buildInfo_descriptors[type].size());
			m_samplerRedirects = SBindingRedirect(buildInfo_samplers.size());

			auto buildRedirect = [](SBindingRedirect& redirect, core::vector<SBindingRedirectBuildInfo>& info)
			{
				std::sort(info.begin(), info.end());

				for (size_t i = 0u; i < info.size(); ++i)
				{
					redirect.bindings[i] = info[i].binding;
					redirect.offsets[i] = info[i].count;
				}

				std::exclusive_scan(redirect.offsets, redirect.offsets + info.size(), redirect.offsets, 0u);
			};

			for (auto type = 0u; type < asset::EDT_COUNT; ++type)
				buildRedirect(m_descriptorRedirects[type], buildInfo_descriptors[type]);
			buildRedirect(m_samplerRedirects, buildInfo_samplers);

			size_t bndCount = _end-_begin;
			size_t immSamplerCount = 0ull;
			for (size_t i = 0ull; i < bndCount; ++i)
			{
				const auto& bnd = _begin[i];
				if (bnd.type == EDT_COMBINED_IMAGE_SAMPLER)
				{
					if (bnd.samplers)
						immSamplerCount += bnd.count;
				}
			}
			m_samplers = immSamplerCount ? core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<sampler_type> > >(immSamplerCount) : nullptr;

			size_t immSamplersOffset = 0u;
			for (size_t i = 0ull; i < bndCount; ++i)
			{
				auto& bnd_out = m_bindings->operator[](i);
				const auto& bnd_in = _begin[i];

				bnd_out.binding = bnd_in.binding;
				bnd_out.type = bnd_in.type;
				bnd_out.count = bnd_in.count;
				bnd_out.stageFlags = bnd_in.stageFlags;
				bnd_out.samplers = nullptr;
				if (bnd_in.type==EDT_COMBINED_IMAGE_SAMPLER && bnd_in.samplers)
				{
                    ++immSamplersOffset;//add 1 so that bnd_out.samplers is never 0/nullptr when the binding SHOULD have imm samplers
                    //otherwise if (bnd.samplers) won't work
					bnd_out.samplers = reinterpret_cast<const core::smart_refctd_ptr<sampler_type>*>(immSamplersOffset);
                    --immSamplersOffset;//going back to prev state
					for (uint32_t s = 0ull; s < bnd_in.count; ++s)
						m_samplers->operator[](immSamplersOffset+s) = bnd_in.samplers[s];
					immSamplersOffset += bnd_in.count;
				}
			}

			std::fill_n(m_descriptorCount, EDT_COUNT, 0ull);
            if (m_bindings)
            {
                for (size_t i = 0ull; i < m_bindings->size(); ++i)
                {
                    auto& bnd = m_bindings->operator[](i);

                    static_assert(sizeof(size_t) == sizeof(bnd.samplers), "Bad reinterpret_cast!");
                    if (bnd.type == EDT_COMBINED_IMAGE_SAMPLER && bnd.samplers)
                        bnd.samplers = m_samplers->data() + reinterpret_cast<size_t>(bnd.samplers) - 1ull;

					m_descriptorCount[bnd.type] += bnd.count;
                }

                // TODO: check for overlapping bindings (bad `SBinding` definitions)
                std::sort(m_bindings->begin(), m_bindings->end());
            }
		}
		virtual ~IDescriptorSetLayout() = default;

		core::smart_refctd_dynamic_array<SBinding> m_bindings;
		core::smart_refctd_dynamic_array<core::smart_refctd_ptr<sampler_type>> m_samplers;

	public:
		bool isIdenticallyDefined(const IDescriptorSetLayout<sampler_type>* _other) const
		{
			if (!_other || getBindings().size()!=_other->getBindings().size())
				return false;

			const size_t cnt = getBindings().size();
			const SBinding* lhs = getBindings().begin();
			const SBinding* rhs = _other->getBindings().begin();
			for (size_t i = 0ull; i < cnt; ++i)
				if (lhs[i] != rhs[i])
					return false;
			return true;
		}

		inline size_t getTotalMutableSamplerCount() const { return m_mutableSamplerCount; }
		inline size_t getTotalDescriptorCount(const E_DESCRIPTOR_TYPE type) const { return m_descriptorCount[type]; }

		core::SRange<const SBinding> getBindings() const { return {m_bindings->data(), m_bindings->data()+m_bindings->size()}; }

		inline uint32_t getBindingCount(const E_DESCRIPTOR_TYPE type) const { return m_descriptorRedirects[type].count; }
		inline uint32_t* getBindingStorage(const E_DESCRIPTOR_TYPE type) const { return m_descriptorRedirects[type].bindings.get(); }
		inline uint32_t* getBindingOffsetStorage(const E_DESCRIPTOR_TYPE type) const { return m_descriptorRedirects[type].offsets; }

		inline uint32_t getDescriptorOffset(const E_DESCRIPTOR_TYPE type, const uint32_t binding) const{ return m_descriptorRedirects[type][binding]; }
		inline uint32_t getMutableSamplerOffset(const uint32_t binding) const { return m_samplerRedirects[binding]; }

	protected:
		// Maps a binding number to a local (to descriptor set layout) offset, for a given descriptor type.
		SBindingRedirect m_descriptorRedirects[asset::EDT_COUNT];
		SBindingRedirect m_samplerRedirects;

	private:
		size_t m_mutableSamplerCount = 0ull;
		size_t m_descriptorCount[EDT_COUNT];
};

}
}

#endif
