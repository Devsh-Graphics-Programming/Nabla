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
public:
	using sampler_type = SamplerType;

	struct SBinding
	{
		enum class E_CREATE_FLAGS : uint8_t
		{
			ECF_NONE = 0,
			ECF_UPDATE_AFTER_BIND_BIT = 1u << 1,
			ECF_UPDATE_UNUSED_WHILE_PENDING_BIT = 1u << 2,
			ECF_PARTIALLY_BOUND_BIT = 1u << 3
		};

		uint32_t binding;
		E_DESCRIPTOR_TYPE type;
		core::bitflag<E_CREATE_FLAGS> createFlags;
		core::bitflag<IShader::E_SHADER_STAGE> stageFlags;
		uint32_t count;
		// Use this if you want an immutable sampler that is baked into the DS layout itself.
		// If its `nullptr` then the sampler used is mutable and can be specified while writing the image descriptor to a binding while updating the DS.
		const core::smart_refctd_ptr<sampler_type>* samplers;

		bool operator<(const SBinding& rhs) const
		{
			if (binding == rhs.binding)
			{
				// should really assert here
				if (type == rhs.type)
				{
					if (count == rhs.count)
					{
						if (stageFlags.value == rhs.stageFlags.value)
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
						return stageFlags.value < rhs.stageFlags.value;
					}
					return count < rhs.count;
				}
				return type < rhs.type;
			}
			return binding < rhs.binding;
		}
	};

	class CBindingRedirect
	{
	public:
		static constexpr inline uint32_t Invalid = ~0u;

		struct binding_number_t
		{
			inline binding_number_t(const uint32_t d) : data(d) {}
			uint32_t data;
		};

		struct storage_offset_t
		{
			inline storage_offset_t(const uint32_t d) : data(d) {}
			uint32_t data;
		};

		inline uint32_t getBindingCount() const { return m_count; }

		// Returns index into the binding property arrays below (including `m_storageOffsets`), for the given binding number `binding`.
		// Assumes `m_bindingNumbers` is sorted and that there are no duplicate values in it.
		inline uint32_t searchForBinding(const binding_number_t binding) const
		{
			if (!m_bindingNumbers)
				return Invalid;

			assert(m_storageOffsets && (m_count != 0u));

			auto found = std::lower_bound(m_bindingNumbers, m_bindingNumbers + m_count, binding, [](binding_number_t a, binding_number_t b) -> bool {return a.data < b.data; });

			if ((found >= m_bindingNumbers + m_count) || (found->data != binding.data))
				return Invalid;

			const uint32_t foundIndex = found - m_bindingNumbers;
			assert(foundIndex < m_count);
			return foundIndex;
		}

		inline binding_number_t getBindingNumber(const uint32_t index) const
		{
			assert(index < m_count);
			return m_bindingNumbers[index];
		}

		inline core::bitflag<IShader::E_SHADER_STAGE> getStageFlags(const uint32_t index) const
		{
			assert(index < m_count);
			return m_stageFlags[index];
		}

		inline uint32_t getCount(const uint32_t index) const
		{
			assert(index < m_count);
			return (index == 0u) ? m_storageOffsets[index].data : m_storageOffsets[index].data - m_storageOffsets[index - 1].data;
		}

		inline storage_offset_t getStorageOffset(const uint32_t index) const
		{
			assert(index < m_count);
			return (index == 0u) ? 0u : m_storageOffsets[index - 1];
		}


		// The follwoing are merely convienience functions for one off use.
		// If you already have an index (the result of `searchForBinding`) lying around use the above functions for quick lookups.

		inline core::bitflag<IShader::E_SHADER_STAGE> getStageFlags(const binding_number_t binding) const
		{
			const auto index = searchForBinding(binding);
			if (index == Invalid)
				return Invalid;

			return getStageFlags(index);
		}

		inline uint32_t getCount(const binding_number_t binding) const
		{
			const auto index = searchForBinding(binding);
			if (index == Invalid)
				return Invalid;

			return getDescriptorCount(index);
		}

		inline storage_offset_t getStorageOffset(const binding_number_t binding) const
		{
			const auto index = searchForBinding(binding);
			if (index == Invalid)
				return Invalid;

			return getStorageOffset(index);
		}

		inline uint32_t getTotalCount() const { return (m_count == 0ull) ? 0u : m_storageOffsets[m_count - 1].data; }

	private:
		friend class IDescriptorSetLayout;
		struct SBuildInfo
		{
			uint32_t binding;
			core::bitflag<typename SBinding::E_CREATE_FLAGS> createFlags;
			core::bitflag<IShader::E_SHADER_STAGE> stageFlags;
			uint32_t count;

			inline bool operator< (const SBuildInfo& other) const { return binding < other.binding; }
		};

		inline CBindingRedirect() = default;

		CBindingRedirect(core::vector<SBuildInfo>&& info) : m_count(static_cast<uint32_t>(info.size()))
		{
			if (m_count <= 0)
				return;

			init();

			std::sort(info.begin(), info.end());

			for (size_t i = 0; i < info.size(); ++i)
			{
				m_bindingNumbers[i].data = info[i].binding;
				m_createFlags[i] = info[i].createFlags;
				m_stageFlags[i] = info[i].stageFlags;
				m_storageOffsets[i].data = info[i].count;
			}

			std::inclusive_scan(m_storageOffsets, m_storageOffsets + m_count, m_storageOffsets,
				[](storage_offset_t a, storage_offset_t b) -> storage_offset_t { return storage_offset_t{ a.data + b.data }; }, storage_offset_t{ 0u });
		}

		inline void init()
		{
			const size_t requiredMemSize = getRequiredMemorySize();
			m_data = std::make_unique<uint8_t[]>(requiredMemSize);
			{
				assert(m_count > 0);

				uint64_t offset = 0ull;

				m_bindingNumbers = reinterpret_cast<binding_number_t*>(m_data.get() + offset);
				offset += m_count * sizeof(binding_number_t);

				m_createFlags = reinterpret_cast<core::bitflag<typename SBinding::E_CREATE_FLAGS>*>(m_data.get() + offset);
				offset += m_count * sizeof(core::bitflag<typename SBinding::E_CREATE_FLAGS>);

				m_stageFlags = reinterpret_cast<core::bitflag<IShader::E_SHADER_STAGE>*>(m_data.get() + offset);
				offset += m_count * sizeof(core::bitflag<IShader::E_SHADER_STAGE>);

				m_storageOffsets = reinterpret_cast<storage_offset_t*>(m_data.get() + offset);
				offset += m_count * sizeof(storage_offset_t);

				assert(offset == requiredMemSize);
			}
		}

		inline size_t getRequiredMemorySize() const
		{
			const size_t result = m_count * (
				sizeof(binding_number_t) +
				sizeof(core::bitflag<typename SBinding::E_CREATE_FLAGS>) +
				sizeof(core::bitflag<IShader::E_SHADER_STAGE>) +
				sizeof(storage_offset_t));
			return result;
		}

		friend class ICPUDescriptorSetLayout;
		inline CBindingRedirect clone() const
		{
			CBindingRedirect result;
			result.m_count = m_count;

			if (result.m_count > 0)
			{
				result.init();
				memcpy(result.m_data.get(), m_data.get(), getRequiredMemorySize());
			}

			return result;
		}

		inline size_t conservativeSizeEstimate() const { return getRequiredMemorySize() + sizeof(*this); }

		uint32_t m_count = 0u;

		binding_number_t* m_bindingNumbers = nullptr;
		core::bitflag<typename SBinding::E_CREATE_FLAGS>* m_createFlags = nullptr;
		core::bitflag<IShader::E_SHADER_STAGE>* m_stageFlags = nullptr;
		storage_offset_t* m_storageOffsets = nullptr;

		std::unique_ptr<uint8_t[]> m_data = nullptr;
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

	IDescriptorSetLayout(const SBinding* const _begin, const SBinding* const _end)
	{
		core::vector<CBindingRedirect::SBuildInfo> buildInfo_descriptors[asset::EDT_COUNT];
		core::vector<CBindingRedirect::SBuildInfo> buildInfo_immutableSamplers;
		core::vector<CBindingRedirect::SBuildInfo> buildInfo_mutableSamplers;

		for (auto b = _begin; b != _end; ++b)
		{
			buildInfo_descriptors[b->type].emplace_back(b->binding, b->createFlags, b->stageFlags, b->count);

			if (b->type == EDT_COMBINED_IMAGE_SAMPLER)
			{
				if (b->samplers)
					buildInfo_immutableSamplers.emplace_back(b->binding, b->createFlags, b->stageFlags, b->count);
				else
					buildInfo_mutableSamplers.emplace_back(b->binding, b->createFlags, b->stageFlags, b->count);
			}
		}

		for (auto type = 0u; type < asset::EDT_COUNT; ++type)
			m_descriptorRedirects[type] = CBindingRedirect(std::move(buildInfo_descriptors[type]));

		m_immutableSamplerRedirect = CBindingRedirect(std::move(buildInfo_immutableSamplers));
		m_mutableSamplerRedirect = CBindingRedirect(std::move(buildInfo_mutableSamplers));

		const uint32_t immutableSamplerCount = m_immutableSamplerRedirect.getTotalCount();
		m_samplers = immutableSamplerCount ? core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<sampler_type>>>(immutableSamplerCount) : nullptr;

		for (auto b = _begin; b != _end; ++b)
		{
			if (b->type == EDT_COMBINED_IMAGE_SAMPLER && b->samplers)
			{
				const auto localOffset = m_immutableSamplerRedirect.getStorageOffset(CBindingRedirect::binding_number_t(b->binding)).data;
				assert(localOffset != m_immutableSamplerRedirect.Invalid);

				auto* dst = m_samplers->begin() + localOffset;
				std::copy_n(b->samplers, b->count, dst);
			}
		}
#if 0
		size_t bndCount = _end-_begin;
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

        if (m_bindings)
        {
            for (size_t i = 0ull; i < m_bindings->size(); ++i)
            {
                auto& bnd = m_bindings->operator[](i);

                static_assert(sizeof(size_t) == sizeof(bnd.samplers), "Bad reinterpret_cast!");
                if (bnd.type == EDT_COMBINED_IMAGE_SAMPLER && bnd.samplers)
                    bnd.samplers = m_samplers->data() + reinterpret_cast<size_t>(bnd.samplers) - 1ull;
            }

            // TODO: check for overlapping bindings (bad `SBinding` definitions)
            std::sort(m_bindings->begin(), m_bindings->end());
        }
#endif
	}

	virtual ~IDescriptorSetLayout() = default;

	core::smart_refctd_dynamic_array<core::smart_refctd_ptr<sampler_type>> m_samplers = nullptr;

public:
	bool isIdenticallyDefined(const IDescriptorSetLayout<sampler_type>* _other) const
	{
		if (!_other || getTotalBindingCount() != _other->getTotalBindingCount())
			return false;

		for (uint32_t t = 0u; t < EDT_COUNT; ++t)
		{
			const auto& lhs = m_descriptorRedirects[t];
			const auto& rhs = _other->m_descriptorRedirects[t];

			const auto bindingCount = lhs.getBindingCount();
			assert(bindingCount==rhs.getBindingCount());

			for (uint32_t i = 0u; i < bindingCount; ++i)
			{
				const bool equal = (lhs.m_bindingNumbers[i].data == rhs.m_bindingNumbers[i].data) && (lhs.m_createFlags[i].value == rhs.m_createFlags[i].value) && (lhs.m_stageFlags[i].value == rhs.m_stageFlags[i].value) && (lhs.getCount(i) == rhs.getCount(i));
				if (!equal)
					return false;
			}
		}

		if (!m_samplers && !_other->m_samplers)
		{
			return true;
		}
		else if (!m_samplers || !_other->m_samplers)
		{
			return false;
		}
		else
		{
			const auto samplerCount = m_samplers->size();
			assert(samplerCount == _other->m_samplers->size());

			for (uint32_t i = 0u; i < samplerCount; ++i)
			{
				if (m_samplers->begin()[i] != _other->m_samplers->begin()[i])
					return false;
			}
			return true;
		}
	}

	// TODO(achal): I'm not sure, should I keep these around?
	inline uint32_t getTotalMutableSamplerCount() const { return m_mutableSamplerRedirect.getTotalCount(); }
	inline uint32_t getTotalDescriptorCount(const E_DESCRIPTOR_TYPE type) const { return m_descriptorRedirects[type].getTotalCount(); }

	inline uint32_t getTotalBindingCount() const
	{
		uint32_t result = 0u;
		for (uint32_t t = 0; t < EDT_COUNT; ++t)
			result += m_descriptorRedirects[t].getBindingCount();

		return result;
	}

	inline const CBindingRedirect& getDescriptorRedirect(const E_DESCRIPTOR_TYPE type) const { return m_descriptorRedirects[type]; }
	inline const CBindingRedirect& getImmutableSamplerRedirect() const { return m_immutableSamplerRedirect; }
	inline const CBindingRedirect& getMutableSamplerRedirect() const { return m_mutableSamplerRedirect; }

protected:
	// Maps a binding number to a local (to descriptor set layout) offset, for a given descriptor type.
	CBindingRedirect m_descriptorRedirects[asset::EDT_COUNT];
	CBindingRedirect m_immutableSamplerRedirect;
	CBindingRedirect m_mutableSamplerRedirect;
};

}
}

#endif
