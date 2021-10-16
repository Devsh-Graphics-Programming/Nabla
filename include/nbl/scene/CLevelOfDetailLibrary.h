// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_C_LEVEL_OF_DETAIL_LIBRARY_H_INCLUDED__
#define __NBL_SCENE_C_LEVEL_OF_DETAIL_LIBRARY_H_INCLUDED__

#include "nbl/scene/ILevelOfDetailLibrary.h"

namespace nbl::scene
{

template<typename LoDChoiceParams=ILevelOfDetailLibrary::DefaultLoDChoiceParams, template<class...> class allocator=core::allocator>
class CLevelOfDetailLibrary : public ILevelOfDetailLibrary
{
	public:
		struct NBL_FORCE_EBO LoDInfo : AlignBase
		{
			LoDInfo() : drawcallInfoCount(0u), totalDrawcallBoneCount(0u), choiceParams() {}
			LoDInfo(const uint16_t drawcallCount, const LoDChoiceParams& _choiceParams, const core::aabbox3df& aabb)
				: drawcallInfoCount(drawcallCount), totalDrawcallBoneCount(0u), choiceParams(_choiceParams)
			{
				std::copy_n(&aabb.MinEdge.X,3u,aabbMin);
				std::copy_n(&aabb.MaxEdge.X,3u,aabbMax);
			}

			inline bool isValid(const LoDInfo& previous) const
			{
				return choiceParams<previous.choiceParams;
			}

			static inline uint32_t getSizeInUvec4(uint32_t drawcallCount)
			{
				return (offsetof(LoDInfo,drawcallInfos[0])+sizeof(DrawcallInfo)*drawcallCount-1u)/alignof(LoDInfo)+1u;
			}

			float aabbMin[3];
			uint16_t drawcallInfoCount;
			// sum of all bone counts for all draws in this LoD
			uint16_t totalDrawcallBoneCount;
			float aabbMax[3];
			LoDChoiceParams choiceParams;
			DrawcallInfo drawcallInfos[1];
			static_assert(alignof(LoDChoiceParams)==alignof(uint32_t));
		};

        static inline core::smart_refctd_ptr<CLevelOfDetailLibrary> create(const ImplicitBufferCreationParameters& params, allocator<uint8_t>&& _alloc=allocator<uint8_t>())
        {
			assert(params.tableCapacity && params.lodCapacity && params.drawcallCapacity);
			const uint32_t tableBufferSize = params.tableCapacity*(sizeof(LoDTableInfo)+(params.lodCapacity-1u)/params.tableCapacity*sizeof(uint32_t)+alignof(LoDTableInfo));
			const uint32_t lodBufferSize = params.lodCapacity*(sizeof(LoDInfo)+(params.drawcallCapacity-1u)/params.lodCapacity*sizeof(uint32_t)+alignof(LoDInfo));

			ExplicitBufferCreationParameters explicitParams;
			{
				static_cast<CreationParametersBase&>(explicitParams) = params;

				video::IGPUBuffer::SCreationParams bufferParams;
				bufferParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
				explicitParams.lodTableInfoBuffer = {0ull,tableBufferSize,params.device->createDeviceLocalGPUBufferOnDedMem(bufferParams,tableBufferSize)};
				explicitParams.lodTableInfoBuffer.buffer->setObjectDebugName("LoD Table Infos");
				explicitParams.lodInfoBuffer = {0ull,lodBufferSize,params.device->createDeviceLocalGPUBufferOnDedMem(bufferParams,lodBufferSize)};
				explicitParams.lodInfoBuffer.buffer->setObjectDebugName("LoD Infos");
			}
			return create(std::move(explicitParams),std::move(_alloc));
		}
        static inline core::smart_refctd_ptr<CLevelOfDetailLibrary> create(ExplicitBufferCreationParameters&& params, allocator<uint8_t>&& _alloc=allocator<uint8_t>())
        {
			if (!params.device || !params.lodTableInfoBuffer.isValid() || !params.lodInfoBuffer.isValid())
				return nullptr;

			const auto allocatorReservedSize = computeReservedSize(params.lodTableInfoBuffer.size,params.lodInfoBuffer.size);
			auto allocatorReserved = std::allocator_traits<allocator<uint8_t>>::allocate(_alloc,allocatorReservedSize);
			if (!allocatorReserved)
				return nullptr;

			auto* lodl = new CLevelOfDetailLibrary(params.device,std::move(params.lodTableInfoBuffer),std::move(params.lodInfoBuffer),allocatorReserved,std::move(_alloc));
			if (!lodl) // TODO: redo this, allocate the memory for the object, if fail, then dealloc, we cannot free from a moved allocator
				std::allocator_traits<allocator<uint8_t>>::deallocate(_alloc,allocatorReserved,allocatorReservedSize);

            return core::smart_refctd_ptr<CLevelOfDetailLibrary>(lodl,core::dont_grab);
        }

		//
		inline bool allocateLoDs(Allocation& params)
		{
			return ILevelOfDetailLibrary::allocateLoDs<LoDInfo>(params);
		}
		//
		inline void freeLoDs(const Allocation& params)
		{
			ILevelOfDetailLibrary::freeLoDs<LoDInfo>(params);
		}

	protected:
		CLevelOfDetailLibrary(video::ILogicalDevice* device, asset::SBufferRange<video::IGPUBuffer>&& _lodTableInfos, asset::SBufferRange<video::IGPUBuffer>&& _lodInfos, void* _allocatorReserved, allocator<uint8_t>&& _alloc)
			: ILevelOfDetailLibrary(device,std::move(_lodTableInfos),std::move(_lodInfos),reinterpret_cast<uint8_t*>(_allocatorReserved),maxInfoCapacity(_lodInfos.size)), m_alloc(std::move(_alloc))
		{
		}
		~CLevelOfDetailLibrary()
		{
            std::allocator_traits<allocator<uint8_t>>::deallocate(
                m_alloc,reinterpret_cast<uint8_t*>(m_allocatorReserved),
				computeReservedSize(m_lodTableInfos.size,m_lodInfos.size)
            );
		}

		static inline uint32_t maxInfoCapacity(const uint64_t lodBufferSize)
		{
			return lodBufferSize/alignof(LoDInfo);
		}
        static inline size_t computeReservedSize(const uint64_t tableBufferSize, const uint64_t lodBufferSize)
        {
            return computeTableReservedSize(tableBufferSize)+core::roundUp(AddressAllocator::reserved_size(1u,maxInfoCapacity(lodBufferSize),1u),16u);
        }

		allocator<uint8_t> m_alloc;
};


} // end namespace nbl::scene

#endif

