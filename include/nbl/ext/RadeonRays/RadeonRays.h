// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_EXT_RADEON_RAYS_H_INCLUDED__
#define __NBL_EXT_RADEON_RAYS_H_INCLUDED__

#include "irrlicht.h"

#define RR_STATIC_LIBRARY
#define USE_OPENCL
#include "../radeonrays/RadeonRays/include/radeon_rays_cl.h"
#undef USE_OPENCL
#undef RR_STATIC_LIBRARY


namespace nbl
{
namespace ext
{
namespace RadeonRays
{


class Manager final : public core::IReferenceCounted
{
	public:
		static core::smart_refctd_ptr<Manager> create(video::IVideoDriver* _driver);

		
		std::pair<::RadeonRays::Buffer*,cl_mem> linkBuffer(const video::IGPUBuffer* buffer, cl_mem_flags access);
		inline void deleteRRBuffer(::RadeonRays::Buffer* buffer)
		{
			rr->DeleteBuffer(buffer);
		}


		using MeshBufferRRShapeCache = core::unordered_map<const asset::ICPUMeshBuffer*,::RadeonRays::Shape*>;
#ifdef TODO
		using MeshNodeRRInstanceCache = core::unordered_map<scene::IMeshSceneNode*, core::smart_refctd_dynamic_array<::RadeonRays::Shape*> >;
#endif

		template<typename Iterator>
		inline void makeRRShapes(MeshBufferRRShapeCache& shapeCache, Iterator _begin, Iterator _end)
		{
			shapeCache.reserve(std::distance(_begin,_end));

			uint32_t maxIndexCount = 0u;
			for (auto it=_begin; it!=_end; it++)
			{
				auto* mb = static_cast<nbl::asset::ICPUMeshBuffer*>(*it);
				auto found = shapeCache.find(mb);
				if (found!=shapeCache.end())
					continue;
				shapeCache.insert({mb,nullptr});


				auto posAttrID = mb->getPositionAttributeIx();
				auto format = mb->getAttribFormat(posAttrID);
				assert(format==asset::EF_R32G32B32A32_SFLOAT||format==asset::EF_R32G32B32_SFLOAT);

				auto pType = mb->getPipeline()->getPrimitiveAssemblyParams().primitiveType;
				switch (pType)
				{
					case asset::EPT_TRIANGLE_STRIP:
						maxIndexCount = core::max((mb->getIndexCount()-2u)/3u, maxIndexCount);
						break;
					case asset::EPT_TRIANGLE_FAN:
						maxIndexCount = core::max(((mb->getIndexCount()-1u)/2u)*3u, maxIndexCount);
						break;
					case asset::EPT_TRIANGLE_LIST:
						maxIndexCount = core::max(mb->getIndexCount(), maxIndexCount);
						break;
					default:
						assert(false);
				}
			}

			if (maxIndexCount ==0u)
				return;

			auto* indices = new int32_t[maxIndexCount];
			for (auto it=_begin; it!=_end; it++)
				makeShape(shapeCache,static_cast<nbl::asset::ICPUMeshBuffer*>(*it),indices);
			delete[] indices;
		}

		template<typename Iterator>
		inline void deleteShapes(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
				rr->DeleteShape(std::get<::RadeonRays::Shape*>(*it));
		}

#ifdef TODO
		template<typename Iterator>
		inline void makeRRInstances(MeshNodeRRInstanceCache& instanceCache, const MeshBufferRRShapeCache& shapeCache,
									asset::IAssetManager* _assetManager, Iterator _begin, Iterator _end, const int32_t* _id_begin=nullptr)
		{
			core::unordered_map<const video::IGPUMeshBuffer*,MeshBufferRRShapeCache::value_type> GPU2CPUTable;
			GPU2CPUTable.reserve(shapeCache.size());
			for (auto record : shapeCache)
			{
				auto gpumesh = dynamic_cast<video::IGPUMeshBuffer*>(_assetManager->findGPUObject(record.first).get());
				if (!gpumesh)
					continue;

				GPU2CPUTable.insert({gpumesh,record});
			}

			auto* id_it = _id_begin;
			for (auto it=_begin; it!=_end; it++,id_it++)
			{
				nbl::scene::IMeshSceneNode* node = *it;
				makeInstance(instanceCache,GPU2CPUTable,node,_id_begin ? id_it:nullptr);
			}
		}

		template<typename Iterator>
		inline void attachInstances(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
			{
				auto* arr = std::get<core::smart_refctd_dynamic_array<::RadeonRays::Shape*> >(*it).get();
				for (auto it2 = arr->begin(); it2 != arr->end(); it2++)
					rr->AttachShape(*it2);
			}
		}

		template<typename Iterator>
		inline void detachInstances(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
			{
				auto* arr = std::get<core::smart_refctd_dynamic_array<::RadeonRays::Shape*> >(*it).get();
				for (auto it2 = arr->begin(); it2 != arr->end(); it2++)
					rr->DetachShape(*it2);
			}
		}

		template<typename Iterator>
		inline void deleteInstances(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
			{
				auto* arr = std::get<core::smart_refctd_dynamic_array<::RadeonRays::Shape*> >(*it).get();
				for (auto it2=arr->begin(); it2!=arr->end(); it2++)
					rr->DeleteShape(*it2);
			}
		}


		inline void update(const MeshNodeRRInstanceCache& instances)
		{
			bool needToCommit = false;
			for (const auto& instance : instances)
			{
				auto absoluteTForm = core::matrix3x4SIMD().set(instance.first->getAbsoluteTransformation());
				auto* shapes = instance.second.get();

				// check if moved
				{
					core::matrix4SIMD oldTForm,dummy;
					shapes->operator[](0)->GetTransform(reinterpret_cast<::RadeonRays::matrix&>(oldTForm),reinterpret_cast<::RadeonRays::matrix&>(dummy));
					if (absoluteTForm==oldTForm.extractSub3x4())
						continue;
				}

				needToCommit = true;
				core::matrix4SIMD world(absoluteTForm);

				core::matrix3x4SIMD tmp;
				absoluteTForm.getInverse(tmp);
				core::matrix4SIMD worldinv(tmp);

				for (auto it=shapes->begin(); it!=shapes->end(); it++)
					(*it)->SetTransform(reinterpret_cast<::RadeonRays::matrix&>(world),reinterpret_cast<::RadeonRays::matrix&>(worldinv));
			}

			if (needToCommit)
				rr->Commit();
		}
#endif

		inline bool hasImplicitCL2GLSync() const { return m_automaticOpenCLSync; }


		inline auto* getRadeonRaysAPI() {return rr;}

		inline cl_command_queue getCLCommandQueue() { return commandQueue; }

	protected:
		Manager(video::IVideoDriver* _driver, cl_context context, bool automaticOpenCLSync);
		~Manager();

		void makeShape(MeshBufferRRShapeCache& shapeCache, const asset::ICPUMeshBuffer* mb, int32_t* indices);
#ifdef TODO
		void makeInstance(	MeshNodeRRInstanceCache& instanceCache,
							const core::unordered_map<const video::IGPUMeshBuffer*,MeshBufferRRShapeCache::value_type>& GPU2CPUTable,
							scene::IMeshSceneNode* node, const int32_t* id_it);
#endif

		video::IVideoDriver* driver;
		::RadeonRays::IntersectionApi* rr;
		cl_context m_context;
		cl_command_queue commandQueue;
		bool m_automaticOpenCLSync;
};

}
}
}

#endif