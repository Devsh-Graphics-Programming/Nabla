// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED_
#define _NBL_ASSET_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED_


#include "nbl/core/declarations.h"
#include "nbl/core/SRange.h"

#include "nbl/asset/IShader.h"


namespace nbl::asset
{

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

	@see IReferenceCounted
*/

template<typename SamplerType>
class IDescriptorSetLayout : public virtual core::IReferenceCounted  // TODO: try to remove this inheritance and see what happens
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
			ECF_PARTIALLY_BOUND_BIT = 1u << 3,
			ECF_VARIABLE_DESCRIPTOR_COUNT_BIT = 1u << 4
		};

		uint32_t binding;
		IDescriptor::E_TYPE type;
		core::bitflag<E_CREATE_FLAGS> createFlags;
		core::bitflag<IShader::E_SHADER_STAGE> stageFlags;
		uint32_t count;
		// Use this if you want an immutable sampler that is baked into the DS layout itself.
		// If its `nullptr` then the sampler used is mutable and can be specified while writing the image descriptor to a binding while updating the DS.
		const core::smart_refctd_ptr<sampler_type>* samplers;
	};

	// Maps a binding to a local (to descriptor set layout) offset.
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

		struct storage_range_index_t
		{
			inline storage_range_index_t(const uint32_t d) : data(d) {}
			uint32_t data;
		};

		inline uint32_t getBindingCount() const { return m_count; }

		// Returns index into the binding property arrays below (including `m_storageOffsets`), for the given binding number `binding`.
		// Assumes `m_bindingNumbers` is sorted and that there are no duplicate values in it.
		inline storage_range_index_t findBindingStorageIndex(const binding_number_t binding) const
		{
			if (!m_bindingNumbers)
				return { Invalid };

			assert(m_storageOffsets && (m_count != 0u));

			auto found = std::lower_bound(m_bindingNumbers, m_bindingNumbers + m_count, binding, [](binding_number_t a, binding_number_t b) -> bool {return a.data < b.data; });

			if ((found >= m_bindingNumbers + m_count) || (found->data != binding.data))
				return { Invalid };

			const uint32_t foundIndex = found - m_bindingNumbers;
			assert(foundIndex < m_count);
			return { foundIndex };
		}

		inline binding_number_t getBinding(const storage_range_index_t index) const
		{
			assert(index.data < m_count);
			return m_bindingNumbers[index.data];
		}

		inline core::bitflag<IShader::E_SHADER_STAGE> getStageFlags(const storage_range_index_t index) const
		{
			assert(index.data < m_count);
			return m_stageFlags[index.data];
		}

		inline core::bitflag<typename SBinding::E_CREATE_FLAGS> getCreateFlags(const storage_range_index_t index) const
		{
			assert(index.data < m_count);
			return m_createFlags[index.data];
		}

		inline uint32_t getCount(const storage_range_index_t index) const
		{
			assert(index.data < m_count);
			return (index.data == 0u) ? m_storageOffsets[index.data].data : m_storageOffsets[index.data].data - m_storageOffsets[index.data - 1].data;
		}

		inline storage_offset_t getStorageOffset(const storage_range_index_t index) const
		{
			assert(index.data < m_count);
			return (index.data == 0u) ? 0u : m_storageOffsets[index.data - 1];
		}

		// The following are merely convenience functions for one off use.
		// If you already have an index (the result of `findBindingStorageIndex`) lying around use the above functions for quick lookups, and to avoid unnecessary binary searches.

		inline core::bitflag<IShader::E_SHADER_STAGE> getStageFlags(const binding_number_t binding) const
		{
			const auto index = findBindingStorageIndex(binding);
			if (index == Invalid)
				return IShader::ESS_UNKNOWN;

			return getStageFlags(index);
		}

		inline uint32_t getCount(const binding_number_t binding) const
		{
			const auto index = findBindingStorageIndex(binding);
			if (index.data == Invalid)
				return 0;

			return getCount(index);
		}

		inline storage_offset_t getStorageOffset(const binding_number_t binding) const
		{
			const auto index = findBindingStorageIndex(binding);
			if (index.data == Invalid)
				return { Invalid };

			return getStorageOffset(index);
		}

		inline uint32_t getTotalCount() const { return (m_count == 0ull) ? 0u : m_storageOffsets[m_count - 1].data; }

	private:
		// error C2248 : 'nbl::asset::IDescriptorSetLayout<nbl::video::IGPUSampler>::CBindingRedirect::CBindingRedirect'
		// : cannot access private member declared in class 'nbl::asset::IDescriptorSetLayout<nbl::video::IGPUSampler>::CBindingRedirect'
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

				// Allocations ordered from fattest alignment to smallest alignment, because there could be problem on ARM.
				m_bindingNumbers = reinterpret_cast<binding_number_t*>(m_data.get() + offset);
				offset += m_count * sizeof(binding_number_t);
				assert(core::is_aligned_ptr(m_bindingNumbers));

				assert(alignof(core::bitflag<IShader::E_SHADER_STAGE>) <= alignof(decltype(m_bindingNumbers[0])));

				m_stageFlags = reinterpret_cast<core::bitflag<IShader::E_SHADER_STAGE>*>(m_data.get() + offset);
				offset += m_count * sizeof(core::bitflag<IShader::E_SHADER_STAGE>);
				assert(core::is_aligned_ptr(m_stageFlags));

				assert(alignof(core::bitflag<IShader::E_SHADER_STAGE>) >= alignof(storage_offset_t));

				m_storageOffsets = reinterpret_cast<storage_offset_t*>(m_data.get() + offset);
				offset += m_count * sizeof(storage_offset_t);
				assert(core::is_aligned_ptr(m_storageOffsets));

				m_createFlags = reinterpret_cast<core::bitflag<typename SBinding::E_CREATE_FLAGS>*>(m_data.get() + offset);
				offset += m_count * sizeof(core::bitflag<typename SBinding::E_CREATE_FLAGS>);
				assert(core::is_aligned_ptr(m_createFlags));

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
	static inline void fillBindingsSameType(SBinding* bindings, uint32_t count, IDescriptor::E_TYPE type, const uint32_t* counts=nullptr, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
	{
		for (auto i=0u; i<count; i++)
		{
			bindings[i].binding = i;
			bindings[i].type = type;
			bindings[i].createFlags = SBinding::E_CREATE_FLAGS::ECF_NONE;
			bindings[i].stageFlags = stageAccessFlags ? stageAccessFlags[i]:asset::IShader::ESS_ALL;
			bindings[i].count = counts ? counts[i]:1u;
			bindings[i].samplers = nullptr;
		}
	}

	bool isSubsetOf(const IDescriptorSetLayout<sampler_type>* other) const
	{
		if (!other || getTotalBindingCount() > other->getTotalBindingCount())
			return false;

		for (uint32_t t = 0u; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
		{
			const auto& bindingRedirects = m_descriptorRedirects[t];
			const auto& otherBindingRedirects = other->m_descriptorRedirects[t];

			const uint32_t bindingCnt = bindingRedirects.getBindingCount();
			const uint32_t otherBindingCnt = otherBindingRedirects.getBindingCount();
			if (bindingCnt > otherBindingCnt)
				return false;

			for (uint32_t b = 0u; b < bindingCnt; ++b)
			{
				uint32_t bindingNumber = m_descriptorRedirects[t].m_bindingNumbers[b].data;
				uint32_t otherBindingNumber = CBindingRedirect::Invalid;
				// TODO: std::find instead?
				for (uint32_t ob = 0u; ob < otherBindingCnt; ++ob)
				{
					if (bindingNumber == other->m_descriptorRedirects[t].m_bindingNumbers[ob].data)
					{
						otherBindingNumber = ob;
						break;
					}
				}

				if (otherBindingNumber == CBindingRedirect::Invalid)
					return false;

				
				const auto storageOffset = bindingRedirects.getStorageOffset(bindingRedirects.findBindingStorageIndex(bindingNumber));
				const auto otherStorageOffset = otherBindingRedirects.getStorageOffset(otherBindingRedirects.findBindingStorageIndex(otherBindingNumber));

				// validate counts
				if (storageOffset.data != otherStorageOffset.data)
					return false;

				// TODO[Przemek]: validate samplers	
			}
		}
		return true;
	}

	bool isIdenticallyDefined(const IDescriptorSetLayout<sampler_type>* _other) const
	{
		if (!_other || getTotalBindingCount() != _other->getTotalBindingCount())
			return false;

		auto areRedirectsEqual = [](const CBindingRedirect& lhs, const CBindingRedirect& rhs) -> bool
		{
			const auto memSize = lhs.getRequiredMemorySize();
			if (memSize != rhs.getRequiredMemorySize())
				return false;

			if (std::memcmp(lhs.m_data.get(), rhs.m_data.get(), memSize) != 0)
				return false;

			return true;
		};

		for (uint32_t t = 0u; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
		{
			if (!areRedirectsEqual(m_descriptorRedirects[t], _other->m_descriptorRedirects[t]))
				return false;
		}

		if (!areRedirectsEqual(m_immutableSamplerRedirect, _other->m_immutableSamplerRedirect))
			return false;

		if (!areRedirectsEqual(m_mutableSamplerRedirect, _other->m_mutableSamplerRedirect))
			return false;

		if (m_samplers && _other->m_samplers)
			return std::equal(m_samplers->begin(), m_samplers->end(), _other->m_samplers->begin(), _other->m_samplers->end());
		else
			return !m_samplers && !_other->m_samplers;
	}

	inline uint32_t getTotalMutableSamplerCount() const { return m_mutableSamplerRedirect.getTotalCount(); }
	inline uint32_t getTotalDescriptorCount(const IDescriptor::E_TYPE type) const { return m_descriptorRedirects[static_cast<uint32_t>(type)].getTotalCount(); }

	inline uint32_t getTotalBindingCount() const
	{
		uint32_t result = 0u;
		for (uint32_t t = 0; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
			result += m_descriptorRedirects[t].getBindingCount();

		return result;
	}

	inline const CBindingRedirect& getDescriptorRedirect(const IDescriptor::E_TYPE type) const { return m_descriptorRedirects[static_cast<uint32_t>(type)]; }
	inline const CBindingRedirect& getImmutableSamplerRedirect() const { return m_immutableSamplerRedirect; }
	inline const CBindingRedirect& getMutableSamplerRedirect() const { return m_mutableSamplerRedirect; }

	inline core::SRange<const core::smart_refctd_ptr<sampler_type>> getImmutableSamplers() const
	{
		if (!m_samplers)
			return { nullptr, nullptr };
		
		return { m_samplers->cbegin(), m_samplers->cend() };
	}

protected:
	IDescriptorSetLayout(const std::span<const SBinding> _bindings)
	{
		core::vector<typename CBindingRedirect::SBuildInfo> buildInfo_descriptors[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)];
		core::vector<typename CBindingRedirect::SBuildInfo> buildInfo_immutableSamplers;
		core::vector<typename CBindingRedirect::SBuildInfo> buildInfo_mutableSamplers;

		for (const auto& b : _bindings)
		{
			buildInfo_descriptors[static_cast<uint32_t>(b.type)].emplace_back(b.binding, b.createFlags, b.stageFlags, b.count);

			if (b.type == IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)
			{
				if (b.samplers)
					buildInfo_immutableSamplers.emplace_back(b.binding, b.createFlags, b.stageFlags, b.count);
				else
					buildInfo_mutableSamplers.emplace_back(b.binding, b.createFlags, b.stageFlags, b.count);
			}
		}

		for (auto type = 0u; type < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++type)
			m_descriptorRedirects[type] = CBindingRedirect(std::move(buildInfo_descriptors[type]));

		m_immutableSamplerRedirect = CBindingRedirect(std::move(buildInfo_immutableSamplers));
		m_mutableSamplerRedirect = CBindingRedirect(std::move(buildInfo_mutableSamplers));

		const uint32_t immutableSamplerCount = m_immutableSamplerRedirect.getTotalCount();
		m_samplers = immutableSamplerCount ? core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<sampler_type>>>(immutableSamplerCount) : nullptr;

		for (const auto& b : _bindings)
		{
			if (b.type == IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER && b.samplers)
			{
				const auto localOffset = m_immutableSamplerRedirect.getStorageOffset(typename CBindingRedirect::binding_number_t(b.binding)).data;
				assert(localOffset != m_immutableSamplerRedirect.Invalid);

				auto* dst = m_samplers->begin() + localOffset;
				std::copy_n(b.samplers, b.count, dst);
			}
		}
	}

	virtual ~IDescriptorSetLayout() = default;

	CBindingRedirect m_descriptorRedirects[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)];
	CBindingRedirect m_immutableSamplerRedirect;
	CBindingRedirect m_mutableSamplerRedirect;

	core::smart_refctd_dynamic_array<core::smart_refctd_ptr<sampler_type>> m_samplers = nullptr;
};

}
#endif
