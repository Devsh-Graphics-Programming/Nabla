// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_LEVEL_OF_DETAIL_LIBRARY_H_INCLUDED__
#define __NBL_SCENE_I_LEVEL_OF_DETAIL_LIBRARY_H_INCLUDED__

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/utilities/IDrawIndirectAllocator.h"

namespace nbl::scene
{

class ILevelOfDetailLibrary : public virtual core::IReferenceCounted
{
	public:
		using AddressAllocator = core::GeneralpurposeAddressAllocator<uint32_t>;
		static inline constexpr auto invalid = AddressAllocator::invalid_address;
		//
		struct DefaultLoDChoiceParams
		{
			float distanceSqAtReferenceFoV;

			inline bool operator<(const DefaultLoDChoiceParams& other) const
			{
				return distanceSqAtReferenceFoV<other.distanceSqAtReferenceFoV;
			}
		};
		//
		struct alignas(16) LoDTableInfo
		{
			float aabbMin[3]; 
			uint32_t levelCount;
			float aabbMax[3];
			uint32_t levelInfoOffsets[1];

			static inline uint32_t getSizeInUvec4(uint32_t levelCount)
			{
				return (offsetof(LoDTableInfo,levelInfoOffsets[0])+sizeof(uint32_t)*levelCount-1u)/alignof(LoDTableInfo)+1u;
			}
		};
		// TODO: later template<typename LoDChoiceParams=DefaultLoDChoiceParams>
		using LoDChoiceParams = DefaultLoDChoiceParams;
		struct alignas(16) LoDInfo
		{
			struct alignas(8) DrawcallInfo
			{
				public:
					uint32_t drawcallDWORDOffset; // only really need 27 bits for this
					// TODO: setter for the skinning AABBs
				private:
					uint32_t skinningAABBCountAndOffset;
			};

			static inline uint32_t getSizeInUvec4(uint32_t drawcallCount)
			{
				return (offsetof(LoDInfo,drawcallInfos[0])+sizeof(DrawcallInfo)*drawcallCount-1u)/alignof(LoDInfo)+1u;
			}

			float aabbMin[3];
			uint16_t drawcallInfoCount;
			// sum of all bone counts for all draws in this LoD
			uint16_t totalDrawCallBoneCount;
			float aabbMax[3];
			LoDChoiceParams choiceParams;
			DrawcallInfo drawcallInfos[1];
			static_assert(alignof(LoDChoiceParams)==alignof(uint32_t));
		};

        static inline core::smart_refctd_ptr<ILevelOfDetailLibrary> create(core::smart_refctd_ptr<video::ILogicalDevice>&& _device, const uint32_t tableCapacity, const uint32_t lodCapacity, const uint32_t drawcallCapacity)
        {
			assert(tableCapacity && lodCapacity && drawcallCapacity);
			const uint32_t tableBufferSize = tableCapacity*sizeof(LoDTableInfo)+core::roundUp<uint32_t>((lodCapacity-1u)/tableCapacity,alignof(LoDTableInfo))*sizeof(uint32_t);
			const uint32_t lodBufferSize = lodCapacity*sizeof(LoDInfo)+core::roundUp<uint32_t>((drawcallCapacity-1u)/lodCapacity,alignof(LoDInfo))*sizeof(uint32_t);

			video::IGPUBuffer::SCreationParams params;
			params.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
			auto tableBuffer = _device->createDeviceLocalGPUBufferOnDedMem(params,tableBufferSize);
			auto lodBuffer = _device->createDeviceLocalGPUBufferOnDedMem(params,lodBufferSize);
			return create(std::move(_device),{0ull,tableBufferSize,tableBuffer},{0ull,lodBufferSize,lodBuffer});
		}
        static inline core::smart_refctd_ptr<ILevelOfDetailLibrary> create(core::smart_refctd_ptr<video::ILogicalDevice>&& _device, asset::SBufferRange<video::IGPUBuffer>&& _lodTableInfos, asset::SBufferRange<video::IGPUBuffer>&& _lodInfos)
        {
			if (!_lodTableInfos.isValid() || !_lodInfos.isValid())
				return nullptr;

			auto* lodl = new ILevelOfDetailLibrary(std::move(_device),std::move(_lodTableInfos),std::move(_lodInfos));
            return core::smart_refctd_ptr<ILevelOfDetailLibrary>(lodl,core::dont_grab);
        }

		//
		struct Allocation
		{
			uint32_t count;
			// must point to an array initialized with `invalid`
			uint32_t* tableUvec4Offsets;
			const uint32_t* levelCounts;
			struct LevelInfoAllocation
			{
				// must point to an array initialized with `invalid`
				uint32_t* levelUvec4Offsets;
				const uint32_t* drawcallCounts;
			};
			LevelInfoAllocation* levelAllocations;
		};
		inline bool allocateLoDs(Allocation& params)
		{
            for (auto i=0u; i<params.count; i++)
            {
                auto& tableOffset = params.tableUvec4Offsets[i];
                if (tableOffset!=invalid)
                    continue;

				const auto levelCount = params.levelCounts[i];
				tableOffset = m_lodTableAllocator.alloc_addr(LoDTableInfo::getSizeInUvec4(levelCount),1u);
                if (tableOffset==invalid)
                    return false;
				for (auto j=0u; j<levelCount; j++)
				{
					auto& levelAlloc = params.levelAllocations[i];
					auto& lodOffset = levelAlloc.levelUvec4Offsets[j];
					if (lodOffset!=invalid)
						continue;

					lodOffset = m_lodInfoAllocator.alloc_addr(LoDInfo::getSizeInUvec4(levelAlloc.drawcallCounts[j]),1u);
					if (lodOffset==invalid)
						return false;
				}
            }
			return true;
		}
		//
		inline void freeLoDs(const Allocation& params)
		{
            for (auto i=0u; i<params.count; i++)
            {
                auto& tableOffset = params.tableUvec4Offsets[i];
                if (tableOffset==invalid)
                    continue;

				const auto levelCount = params.levelCounts[i];
				m_lodTableAllocator.free_addr(tableOffset,LoDTableInfo::getSizeInUvec4(levelCount));
				for (auto j=0u; j<levelCount; j++)
				{
					auto& levelAlloc = params.levelAllocations[i];
					auto& lodOffset = levelAlloc.levelUvec4Offsets[j];
					if (lodOffset==invalid)
						continue;
					
					m_lodInfoAllocator.free_addr(lodOffset,LoDInfo::getSizeInUvec4(levelAlloc.drawcallCounts[j]));
				}
            }
		}
		//
		inline void clear()
		{
			m_lodTableAllocator.reset();
			m_lodInfoAllocator.reset();
		}

		//
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDescriptorSetLayout(video::ILogicalDevice* device)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[DescriptorBindingCount];
			for (auto i=0u; i<DescriptorBindingCount; i++)
			{
				bindings[i].binding = i;
				bindings[i].type = asset::EDT_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
				bindings[i].samplers = nullptr;
			}
			return device->createGPUDescriptorSetLayout(bindings,bindings+DescriptorBindingCount);
		}

		inline const auto getDescriptorSet() const
		{
			return m_ds.get();
		}

		inline const auto& getLodTableInfoBinding() const
		{
			return m_lodTableInfos;
		}

		inline const auto& getLoDTInfoBinding() const
		{
			return m_lodInfos;
		}

	protected:
		ILevelOfDetailLibrary(core::smart_refctd_ptr<video::ILogicalDevice>&& _device, asset::SBufferRange<video::IGPUBuffer>&& _lodTableInfos, asset::SBufferRange<video::IGPUBuffer>&& _lodInfos)
			: m_device(std::move(_device)), m_lodTableInfos(std::move(_lodTableInfos)), m_lodInfos(std::move(_lodInfos))
		{
			auto layout = createDescriptorSetLayout(m_device.get());
			auto pool = m_device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,&layout.get(),&layout.get()+1u);
			m_ds = m_device->createGPUDescriptorSet(pool.get(),std::move(layout));
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet writes[DescriptorBindingCount];
				video::IGPUDescriptorSet::SDescriptorInfo infos[DescriptorBindingCount] =
				{
					m_lodTableInfos,
					m_lodInfos
				};
				for (auto i=0u; i<DescriptorBindingCount; i++)
				{
					writes[i].dstSet = m_ds.get();
					writes[i].binding = i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
					writes[i].info = infos+i;
				}
				m_device->updateDescriptorSets(DescriptorBindingCount,writes,0u,nullptr);
			}
		}
		~ILevelOfDetailLibrary()
		{
			// everything drops itself automatically
		}

		static inline constexpr auto DescriptorBindingCount = 2u;

		AddressAllocator m_lodTableAllocator,m_lodInfoAllocator;
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		asset::SBufferRange<video::IGPUBuffer> m_lodTableInfos,m_lodInfos;
		void* m_allocatorReserved;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_ds;
};


} // end namespace nbl::scene

#endif

