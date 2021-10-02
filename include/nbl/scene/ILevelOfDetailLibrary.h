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
		struct alignas(16) LoDTableInfo
		{
			float aabbMin[3]; 
			uint32_t levelCount;
			float aabbMax[3];
			uint32_t levelInfoOffsets[1];
		};
		struct alignas(16) LoDInfo
		{
			float aabbMin[3];
			uint32_t drawCallInfoCount; // only really need 30 bits for this
			float aabbMax[3];
			float distanceSqAtReferenceFoV; // TODO: template and assert alignof==alignof(uint32_t)
			uint32_t drawCallInfoOffsets[1];
		};
		struct alignas(8) DrawCallInfo
		{
			uint32_t drawCallDWORDOffset; // only really need 27 bits for this
			uint32_t aabbCount : 8u; // should probably make the bit allocation a function of max bone count
			uint32_t aabbOffset : 24u; // could want to give 26 bits here
		};

		//
		struct DefaultLoDParameters
		{
			float distanceSqAtReferenceFoV;

			inline bool operator<(const DefaultLoDParameters& other) const
			{
				return distanceSqAtReferenceFoV<other.distanceSqAtReferenceFoV;
			}
		};
#if 0
        static inline core::smart_refctd_ptr<ILevelOfDetailLibrary> create(core::smart_refctd_ptr<video::ILogicalDevice>&& _device, const uint32_t tableCapacity, const uint32_t lodCapacity, const uint32_t drawCallCapacity)
        {
			const uint32_t tableBufferSize = tableCapacity*sizeof(LoDTableInfo)+core::roundUp<uint32_t>((lodCapacity-1u)/tableCapacity+1u,alignof(LoDTableInfo))*sizeof(uint32_t);
			const uint32_t lodBufferSize = lodCapacity*sizeof(LoDInfo)+core::roundUp<uint32_t>((drawCallCapacity-1u)/lodCapacity+1u,alignof(LoDInfo))*sizeof(uint32_t);
			if (true) // TODO: some checks and validation before creating?
				return nullptr;

			auto* lodl = new ILevelOfDetailLibrary(std::move(_device)/*,std::move(),std::move(),std::move()*/);
            return core::smart_refctd_ptr<ILevelOfDetailLibrary>(lodl,core::dont_grab);
        }

		// TODO: register/deregister drawcalls/lods/tables
		template<typename MeshIterator, typename CullParamsIterator>
		struct RegisterLoDTable
		{
			MeshIterator beginMeshes;
			MeshIterator endMeshes;
			CullParamsIterator beginCullParams;
		};
		template<typename MeshBufferIterator>
		struct RegisterLoD
		{
			MeshBufferIterator beginMeshBuffers;
			MeshBufferIterator endMeshBuffers;
		};

		template<typename MeshBufferIterator>
		draw_call_t registerDrawcalls(MeshBufferIterator begin, MeshBufferIterator end)
		{
			assert(false); // TODO
		}
		template<typename MeshBufferIterator>
		draw_call_t deregisterDrawcalls(MeshBufferIterator begin, MeshBufferIterator end)
		{
			assert(false); // TODO
		}
#endif
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

		inline const auto& getAABBBinding() const
		{
			return m_aabbs;
		}

	protected:
		ILevelOfDetailLibrary(core::smart_refctd_ptr<video::ILogicalDevice>&& _device) : m_device(std::move(_device))
		{
			auto layout = createDescriptorSetLayout(m_device.get());
			auto pool = m_device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,&layout.get(),&layout.get()+1u);
			m_ds = m_device->createGPUDescriptorSet(pool.get(),std::move(layout));
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet writes[DescriptorBindingCount];
				video::IGPUDescriptorSet::SDescriptorInfo infos[DescriptorBindingCount] =
				{
					m_lodTableInfos,
					m_lodInfos,
					m_aabbs
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

		static inline constexpr auto DescriptorBindingCount = 3u;

		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_ds;
		asset::SBufferRange<video::IGPUBuffer> m_lodTableInfos,m_lodInfos,m_aabbs;
};


} // end namespace nbl::scene

#endif

