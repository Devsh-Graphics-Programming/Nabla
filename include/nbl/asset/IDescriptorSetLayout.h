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

		inline uint32_t getBindingCount() const { return count; }

		inline storage_offset_t getStorageOffset(const binding_number_t binding, uint32_t index = Invalid) const
		{
			if (index == Invalid)
			{
				index = searchForBinding(binding);
				if (index == Invalid)
					return Invalid;
			}

			return (index == 0u) ? 0u : storageOffsets[index - 1];
		}

		inline uint32_t getDescriptorCount(const binding_number_t binding, uint32_t index = Invalid) const
		{
			if (index == Invalid)
			{
				index = searchForBinding(binding);
				if (index == Invalid)
					return Invalid;
			}

			return (index == 0u) ? storageOffsets[index].data : storageOffsets[index].data - storageOffsets[index - 1].data;
		}

		inline uint32_t getTotalDescriptorCount() const { return (count == 0ull) ? 0u : storageOffsets[count - 1].data; }

		// TODO(achal): I shouldn't be needing these anymore.
		inline const binding_number_t* getBindingNumbers() const { return bindingNumbers; }
		inline const storage_offset_t* getStorageOffsets() const { return storageOffsets; }

		// Returns index into the binding property arrays below (including `storageOffsets`), for the given binding number `binding`.
		// Assumes `bindingNumbers` is sorted and that there are no duplicate values in it.
		inline uint32_t searchForBinding(const binding_number_t binding) const
		{
			if (!bindingNumbers)
				return Invalid;

			assert(storageOffsets && (count != 0u));

			auto found = std::lower_bound(bindingNumbers, bindingNumbers + count, binding, [](binding_number_t a, binding_number_t b) -> bool {return a.data < b.data; });

			if ((found >= bindingNumbers + count) || (found->data != binding.data))
				return Invalid;

			const uint32_t foundIndex = found - bindingNumbers;
			assert(foundIndex < count);
			return foundIndex;
		}

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

		CBindingRedirect(core::vector<SBuildInfo>&& info) : count(static_cast<uint32_t>(info.size()))
		{
			if (count <= 0)
				return;

			const size_t requiredMemSize = count * (
				sizeof(binding_number_t) +
				sizeof(core::bitflag<typename SBinding::E_CREATE_FLAGS>) +
				sizeof(core::bitflag<IShader::E_SHADER_STAGE>) +
				sizeof(storage_offset_t));

			data = std::make_unique<uint8_t[]>(requiredMemSize);
			dataAllocator = core::LinearAddressAllocator<uint32_t>(nullptr, 0u, 0u, 1u, requiredMemSize);

			bindingNumbers = reinterpret_cast<binding_number_t*>(data.get() + dataAllocator.alloc_addr(count * sizeof(binding_number_t), 1u));
			createFlags = reinterpret_cast<core::bitflag<typename SBinding::E_CREATE_FLAGS>*>(data.get() + dataAllocator.alloc_addr(count * sizeof(core::bitflag<typename SBinding::E_CREATE_FLAGS>), 1u));
			stageFlags = reinterpret_cast<core::bitflag<IShader::E_SHADER_STAGE>*>(data.get() + dataAllocator.alloc_addr(count * sizeof(core::bitflag<IShader::E_SHADER_STAGE>), 1u));
			storageOffsets = reinterpret_cast<storage_offset_t*>(data.get() + dataAllocator.alloc_addr(count * sizeof(storage_offset_t), 1u));

			std::sort(info.begin(), info.end());

			for (size_t i = 0; i < info.size(); ++i)
			{
				bindingNumbers[i].data = info[i].binding;
				createFlags[i] = info[i].createFlags;
				stageFlags[i] = info[i].stageFlags;
				storageOffsets[i].data = info[i].count;
			}

			std::inclusive_scan(storageOffsets, storageOffsets + count, storageOffsets,
				[](storage_offset_t a, storage_offset_t b) -> storage_offset_t { return storage_offset_t{ a.data + b.data }; }, storage_offset_t{ 0u });
		}

		uint32_t count = 0u;

		binding_number_t* bindingNumbers = nullptr;
		core::bitflag<typename SBinding::E_CREATE_FLAGS>* createFlags = nullptr;
		core::bitflag<IShader::E_SHADER_STAGE>* stageFlags = nullptr;
		storage_offset_t* storageOffsets = nullptr;

		std::unique_ptr<uint8_t[]> data = nullptr;
		core::LinearAddressAllocator<uint32_t> dataAllocator;
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

	IDescriptorSetLayout(const SBinding* const _begin, const SBinding* const _end) : 
		m_bindings((_end-_begin) ? core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SBinding>>(_end-_begin) : nullptr)
	{
		core::vector<CBindingRedirect::SBuildInfo> buildInfo_descriptors[asset::EDT_COUNT];
		core::vector<CBindingRedirect::SBuildInfo> buildInfo_samplers;

		for (auto b = _begin; b != _end; ++b)
		{
			buildInfo_descriptors[b->type].emplace_back(b->binding, b->createFlags, b->stageFlags, b->count);
			if (b->type == EDT_COMBINED_IMAGE_SAMPLER && b->samplers == nullptr)
				buildInfo_samplers.emplace_back(b->binding, b->createFlags, b->stageFlags, b->count);
		}

		for (auto type = 0u; type < asset::EDT_COUNT; ++type)
			m_descriptorRedirects[type] = CBindingRedirect(std::move(buildInfo_descriptors[type]));
		m_samplerRedirects = CBindingRedirect(std::move(buildInfo_samplers));

		const uint32_t immutableSamplerCount = m_descriptorRedirects[EDT_COMBINED_IMAGE_SAMPLER].getTotalDescriptorCount() - m_samplerRedirects.getTotalDescriptorCount();
		assert(static_cast<int32_t>(immutableSamplerCount) >= 0);

		m_samplers = immutableSamplerCount ? core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<sampler_type>>>(immutableSamplerCount) : nullptr;

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
	}

	virtual ~IDescriptorSetLayout() = default;

	core::smart_refctd_dynamic_array<SBinding> m_bindings; // TODO(achal): Shoudn't need this anymore.
	core::smart_refctd_dynamic_array<core::smart_refctd_ptr<sampler_type>> m_samplers;

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
				const bool equal = (lhs.bindingNumbers[i].data == rhs.bindingNumbers[i].data) && (lhs.createFlags[i].value == rhs.createFlags[i].value) && (lhs.stageFlags[i].value == rhs.stageFlags[i].value) && (lhs.getDescriptorCount(lhs.bindingNumbers[i], i) == rhs.getDescriptorCount(rhs.bindingNumbers[i], i));
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

	// TODO(achal): I shouldn't be needing these anymore.
	inline size_t getTotalMutableSamplerCount() const { return m_samplerRedirects.getTotalDescriptorCount(); }
	inline size_t getTotalDescriptorCount(const E_DESCRIPTOR_TYPE type) const { return m_descriptorRedirects[type].getTotalDescriptorCount(); }

	core::SRange<const SBinding> getBindings() const { return {m_bindings->data(), m_bindings->data()+m_bindings->size()}; }

	inline uint32_t getTotalBindingCount() const
	{
		uint32_t result = 0u;
		for (uint32_t t = 0; t < EDT_COUNT; ++t)
			result += m_descriptorRedirects[t].getBindingCount();

		return result;
	}

	inline const CBindingRedirect& getDescriptorRedirect(const E_DESCRIPTOR_TYPE type) const { return m_descriptorRedirects[type]; }
	inline const CBindingRedirect& getSamplerRedirect() const { return m_samplerRedirects; }

	// TODO(achal): I shouldn't be needing these anymore.
	inline uint32_t getDescriptorOffset(const E_DESCRIPTOR_TYPE type, const uint32_t binding) const { return m_descriptorRedirects[type].getStorageOffset(binding).data; }
	inline uint32_t getMutableSamplerOffset(const uint32_t binding) const { return m_samplerRedirects.getStorageOffset(binding).data; }

protected:
	// Maps a binding number to a local (to descriptor set layout) offset, for a given descriptor type.
	CBindingRedirect m_descriptorRedirects[asset::EDT_COUNT];
	CBindingRedirect m_samplerRedirects;
};

}
}

#endif
