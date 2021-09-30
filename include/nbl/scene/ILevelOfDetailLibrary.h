// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_LEVEL_OF_DETAIL_LIBRARY_H_INCLUDED__
#define __NBL_SCENE_I_LEVEL_OF_DETAIL_LIBRARY_H_INCLUDED__

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
		// LoD will store a contiguous list of draw_call_t inside itself (past its end)
		using lod_info_t = uint32_t;
		struct alignas(16) DefaultLevelInfo
		{
			float aabbMin[3]; 
			float distanceSqAtReferenceFoV;
			float aabbMax[3];
			uint32_t drawCallCount;
		};

		// TODO: Drawcall struct?
		using draw_call_t = uint32_t;
		struct CullParameters
		{
			float distanceSq;

			inline bool operator<(const CullParameters& other) const
			{
				return distanceSq<other.distanceSq;
			}
		};
		// LoDTable will store a contiguous list of lod_t inside itself (first uint is the count)
		using lod_table_t = uint32_t;

        static inline core::smart_refctd_ptr<ILevelOfDetailLibrary> create(core::smart_refctd_ptr<video::ILogicalDevice>&& _device, const uint32_t tableCapacity, const uint32_t lodCapacity, const uint32_t drawCallCapacity)
        {
			const uint32_t tableBufferSize = tableCapacity*sizeof(LoDTableInfo)+core::roundUp<uint32_t>((lodCapacity-1u)/tableCapacity+1u,alignof(LoDTableInfo))*sizeof(uint32_t);
			const uint32_t lodBufferSize = lodCapacity*sizeof(DefaultLevelInfo)+core::roundUp<uint32_t>((drawCallCapacity-1u)/lodCapacity+1u,alignof(DefaultLevelInfo))*sizeof(uint32_t);
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

	protected:
		ILevelOfDetailLibrary(core::smart_refctd_ptr<video::ILogicalDevice>&& _device) : m_device(std::move(_device))
		{
		}
		~ILevelOfDetailLibrary()
		{
			// everything drops itself automatically
		}

		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
};


} // end namespace nbl::scene

#endif

