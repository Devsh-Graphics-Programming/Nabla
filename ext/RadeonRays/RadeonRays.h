#ifndef __IRR_EXT_RADEON_RAYS_H_INCLUDED__
#define __IRR_EXT_RADEON_RAYS_H_INCLUDED__

#include "irrlicht.h"

#define RR_STATIC_LIBRARY
#define USE_OPENCL
#include "radeonrays/RadeonRays/include/radeon_rays_cl.h"
#undef USE_OPENCL
#undef RR_STATIC_LIBRARY

#include "../../ext/RadeonRays/RadeonRaysIncludeLoader.h"

namespace irr
{
namespace ext
{
namespace RadeonRays
{


class Manager final : public core::IReferenceCounted
{
	public:
		static core::smart_refctd_ptr<Manager> create(video::IVideoDriver* _driver);

		
		::RadeonRays::Buffer* linkBuffer(const video::IGPUBuffer* buffer, cl_mem_flags access);
		inline void deleteRRBuffer(::RadeonRays::Buffer* buffer)
		{
			rr->DeleteBuffer(buffer);
		}


		using MeshBufferRRShapeCache = core::unordered_map<const asset::ICPUMeshBuffer*,::RadeonRays::Shape*>;
		void makeRRShapes(MeshBufferRRShapeCache& shapeCache, const asset::ICPUMeshBuffer** _begin, const asset::ICPUMeshBuffer** _end);

		template<typename Iterator>
		inline void deleteShapes(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
				rr->DeleteShape(std::get<0>(*it));
		}

		using MeshNodeRRInstanceCache = core::unordered_map<scene::IMeshSceneNode*,core::smart_refctd_dynamic_array<::RadeonRays::Shape*> >;
		void makeRRInstances(MeshNodeRRInstanceCache& instanceCache, const MeshBufferRRShapeCache& shapeCache, asset::IAssetManager* _assetManager, scene::IMeshSceneNode** _begin, scene::IMeshSceneNode** _end);

		template<typename Iterator>
		inline void attachInstances(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
			{
				auto* arr = std::get<0>(*it).get();
				for (auto it2 = arr->begin(); it2 != arr->end(); it2++)
					rr->AttachShape(*it2);
			}
		}

		template<typename Iterator>
		inline void detachInstances(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
			{
				auto* arr = std::get<0>(*it).get();
				for (auto it2 = arr->begin(); it2 != arr->end(); it2++)
					rr->DetachShape(*it2);
			}
		}

		template<typename Iterator>
		inline void deleteInstances(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
			{
				auto* arr = std::get<0>(*it).get();
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

		inline RadeonRaysIncludeLoader* getRadeonRaysGLSLIncludes()
		{
			return radeonRaysIncludes.get();
		}


		inline auto* getRadeonRaysAPI() {return rr;}

	protected:
		Manager(video::IVideoDriver* _driver);
		~Manager();

		
		static core::smart_refctd_ptr<RadeonRaysIncludeLoader> radeonRaysIncludes;
		static cl_context context;

		video::IVideoDriver* driver;
		cl_command_queue commandQueue;
		::RadeonRays::IntersectionApi* rr;
};

}
}
}

#endif