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

			static inline float getFoVDilationFactor(const core::matrix4SIMD& proj)
			{
				if (proj.rows[3].w!=0.f)
					return core::nan<float>();
				return abs(proj.rows[0].x*proj.rows[1].y-proj.rows[0].y*proj.rows[1].x)/dot(proj.rows[3],proj.rows[3]).x;
			}
		};
		template<typename InfoType, template<class...> class container=core::vector>
		class InfoContainerAdaptor
		{			
			public:
				struct NBL_FORCE_EBO alignas(InfoType) AlignBase
				{
				};

				inline void reserve(const uint32_t alignmentUnits)
				{
					storage.reserve(alignmentUnits);
				}
				inline InfoType& emplace_back(const uint32_t variableEntryCount)
				{
					const auto oldEnd = storage.size();
					const auto infoSize = InfoType::getSizeInAlignmentUnits(variableEntryCount);
					storage.resize(oldEnd+infoSize);
					return *reinterpret_cast<InfoType*>(storage.data()+oldEnd);
				}
				//
				inline AlignBase* data()
				{
					return storage.data();
				}
				inline const AlignBase* data() const
				{
					return storage.data();
				}

			private:
				container<AlignBase> storage;
		};
		//
		struct alignas(16) LoDTableInfo
		{
			LoDTableInfo() : levelCount(0u) {}
			LoDTableInfo(const uint32_t lodLevelCount) : levelCount(lodLevelCount)
			{
				std::fill_n(aabbMin,3u,FLT_MAX);
				std::fill_n(aabbMax,3u,-FLT_MAX);
			}
			LoDTableInfo(const uint32_t lodLevelCount, const core::aabbox3df& aabb) : levelCount(lodLevelCount)
			{
				std::copy_n(&aabb.MinEdge.X,3u,aabbMin);
				std::copy_n(&aabb.MaxEdge.X,3u,aabbMax);
			}

			static inline uint32_t getSizeInAlignmentUnits(uint32_t levelCount)
			{
				return (offsetof(LoDTableInfo,leveInfoUvec2Offsets[0])+sizeof(uint32_t)*levelCount-1u)/alignof(LoDTableInfo)+1u;
			}

			float aabbMin[3]; 
			uint32_t levelCount;
			float aabbMax[3];
			uint32_t leveInfoUvec2Offsets[1]; // the array isnt really 1-sized, its `levelCount` entries
		};
		struct LoDInfoBase
		{
			LoDInfoBase() : drawcallInfoCount(0u),totalDrawcallBoneCount(0u) {}
			LoDInfoBase(const uint16_t drawcallCount) : drawcallInfoCount(drawcallCount),totalDrawcallBoneCount(0u) {}

			uint16_t drawcallInfoCount;
			// sum of all bone counts for all draws in this LoD
			uint16_t totalDrawcallBoneCount;
		};
		struct alignas(8) DrawcallInfo
		{
			public:
				DrawcallInfo() : aabb(), drawcallDWORDOffset(video::IDrawIndirectAllocator::invalid_draw_range_begin) {}
				DrawcallInfo(const uint32_t _drawcallDWORDOffset)
					: aabb(), drawcallDWORDOffset(_drawcallDWORDOffset) {}
				DrawcallInfo(const uint32_t _drawcallDWORDOffset, const core::aabbox3df& _aabb)
					: aabb(_aabb), drawcallDWORDOffset(_drawcallDWORDOffset) {}

			private:
				core::CompressedAABB aabb;
				uint32_t drawcallDWORDOffset; // only really need 27 bits for this
				uint32_t skinningAABBCountAndOffset = 0u; // TODO
		};

		//
		struct CreationParametersBase
		{
			video::ILogicalDevice* device;
		};
		struct ImplicitBufferCreationParameters : CreationParametersBase
		{
			uint32_t tableCapacity;
			uint32_t lodCapacity;
			uint32_t drawcallCapacity;
		};
		struct ExplicitBufferCreationParameters : CreationParametersBase
		{
			asset::SBufferRange<video::IGPUBuffer> lodTableInfoBuffer;
			asset::SBufferRange<video::IGPUBuffer> lodInfoBuffer;
		};

		//
		struct Allocation
		{
			uint32_t count = 0u;
			// must point to an array initialized with `invalid`
			uint32_t* tableUvec4Offsets = nullptr;
			const uint32_t* levelCounts = nullptr;
			struct LevelInfoAllocation
			{
				// must point to an array initialized with `invalid`
				uint32_t* levelUvec2Offsets = nullptr;
				const uint32_t* drawcallCounts = nullptr;
			};
			LevelInfoAllocation* levelAllocations;
		};
		template<typename LoDInfo>
		inline bool allocateLoDs(Allocation& params)
		{
            for (auto i=0u; i<params.count; i++)
            {
                auto& tableOffset = params.tableUvec4Offsets[i];
                if (tableOffset!=invalid)
                    continue;

				const auto levelCount = params.levelCounts[i];
				tableOffset = m_lodTableAllocator.alloc_addr(LoDTableInfo::getSizeInAlignmentUnits(levelCount),1u);
                if (tableOffset==invalid)
                    return false;
				auto& levelAlloc = params.levelAllocations[i];
				for (auto j=0u; j<levelCount; j++)
				{
					auto& lodOffset = levelAlloc.levelUvec2Offsets[j];
					if (lodOffset!=invalid)
						continue;

					lodOffset = m_lodInfoAllocator.alloc_addr(LoDInfo::getSizeInAlignmentUnits(levelAlloc.drawcallCounts[j]),1u);
					if (lodOffset==invalid)
						return false;
				}
            }
			return true;
		}
		template<typename LoDInfo>
		inline void freeLoDs(const Allocation& params)
		{
            for (auto i=0u; i<params.count; i++)
            {
                auto& tableOffset = params.tableUvec4Offsets[i];
                if (tableOffset==invalid)
                    continue;

				const auto levelCount = params.levelCounts[i];
				m_lodTableAllocator.free_addr(tableOffset,LoDTableInfo::getSizeInAlignmentUnits(levelCount));
				auto& levelAlloc = params.levelAllocations[i];
				for (auto j=0u; j<levelCount; j++)
				{
					auto& lodOffset = levelAlloc.levelUvec2Offsets[j];
					if (lodOffset==invalid)
						continue;
					
					m_lodInfoAllocator.free_addr(lodOffset,LoDInfo::getSizeInAlignmentUnits(levelAlloc.drawcallCounts[j]));
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
				bindings[i].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = asset::IShader::ESS_COMPUTE;
				bindings[i].samplers = nullptr;
			}
			return device->createDescriptorSetLayout(bindings);
		}

		inline const auto getDescriptorSet() const
		{
			return m_ds.get();
		}

		inline const auto& getLodTableInfoBinding() const
		{
			return m_lodTableInfos;
		}

		inline const auto& getLoDInfoBinding() const
		{
			return m_lodInfos;
		}

	protected:
		ILevelOfDetailLibrary(video::ILogicalDevice* device, asset::SBufferRange<video::IGPUBuffer>&& _lodTableInfos, asset::SBufferRange<video::IGPUBuffer>&& _lodInfos, uint8_t* _allocatorReserved, const uint32_t maxInfoCapacity)
			:	m_lodTableAllocator(_allocatorReserved,0u,0u,1u,maxTableCapacity(_lodTableInfos.size),1u),m_lodInfoAllocator(_allocatorReserved+computeTableReservedSize(_lodTableInfos.size),0u,0u,1u,maxInfoCapacity,1u),
				m_allocatorReserved(_allocatorReserved), m_lodTableInfos(std::move(_lodTableInfos)), m_lodInfos(std::move(_lodInfos))
		{
			auto layout = createDescriptorSetLayout(device);
			auto pool = device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,{&layout.get(),1u});
			m_ds = pool->createDescriptorSet(std::move(layout));
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
					writes[i].info = infos+i;
				}
				device->updateDescriptorSets(DescriptorBindingCount,writes,0u,nullptr);
			}
		}
		~ILevelOfDetailLibrary()
		{
			// everything drops itself automatically
		}

		static inline uint32_t maxTableCapacity(const uint64_t tableBufferSize)
		{
			return tableBufferSize/alignof(LoDTableInfo);
		}
        static inline size_t computeTableReservedSize(const uint64_t tableBufferSize)
        {
			return core::roundUp(AddressAllocator::reserved_size(1u,maxTableCapacity(tableBufferSize),1u),_NBL_SIMD_ALIGNMENT);
        }


		static inline constexpr auto DescriptorBindingCount = 2u;

		AddressAllocator m_lodTableAllocator,m_lodInfoAllocator;
		void* m_allocatorReserved;
		asset::SBufferRange<video::IGPUBuffer> m_lodTableInfos,m_lodInfos;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_ds;
};


} // end namespace nbl::scene

#endif

