#ifndef __IRR_EXT_OPTIX_MANAGER_H_INCLUDED__
#define __IRR_EXT_OPTIX_MANAGER_H_INCLUDED__

#include "irrlicht.h"

#include "../src/irr/video/CCUDAHandler.h"

#include "optix.h"

namespace irr
{
namespace ext
{
namespace OptiX
{


class Manager final : public core::IReferenceCounted
{
	public:
		static core::smart_refctd_ptr<Manager> create(video::IVideoDriver* _driver);

		using MappedBuffer = core::unordered_map<const asset::ICPUMeshBuffer*, ::RadeonRays::Shape*>;
		template<typename Iterator>
		OptixTraversableHandle createAccelerationStructure(Iterator _begin, Iterator _end, const OptixAccelBuildOptions& accel_options, uint32_t deviceID=0u, size_t scratchBufferSize=0u, CUdeviceptr scratchBuffer = nullptr)
		{
			auto meshCount = 0u;
			for (auto it=_begin; it!=_end; it++)
			{
				auto* mb = static_cast<video::IGPUMeshBuffer*>(*it);
				auto pipeline = mb->getMeshDataAndFormat();
				auto posbuffer = pipeline->getMappedBuffer(mb->getPositionAttributeIx());
				meshCount++;
			}

			constexpr uint32_t buildStepSize = 256u;
			auto it = _begin;
			for (auto i=0; i<meshCount; i+=buildStepSize)
			{
				const auto oldbegin = it;
				OptixBuildInput buildInputs[buildStepSize] = {};
				OptixAccelBufferSizes buffSizes[buildStepSize] = {};
				for (auto j=0u; it!=_end; it++,j++)
				{
					auto* mb = static_cast<irr::asset::ICPUMeshBuffer*>(*it);
					buildInputs[j].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
					buildInputs[j].triangleArray.vertexBuffers = mb->getPositionAttributeIx();
					buildInputs[j].triangleArray.numVertices = mb->
				}
				optixAccelComputeMemoryUsage(optixContext[deviceID],&accelOptions,buildInputs,std::distance(oldbegin,it),buffSizes);
			}

			return nullptr;
		}
	/*
		using MeshBufferRRShapeCache = core::unordered_map<const asset::ICPUMeshBuffer*,::RadeonRays::Shape*>;
		using MeshNodeRRInstanceCache = core::unordered_map<scene::IMeshSceneNode*, core::smart_refctd_dynamic_array<::RadeonRays::Shape*> >;


		template<typename Iterator>
		inline void makeRRShapes(MeshBufferRRShapeCache& shapeCache, Iterator _begin, Iterator _end)
		{
			shapeCache.reserve(std::distance(_begin,_end));

			uint32_t maxIndexCount = 0u;
			for (auto it=_begin; it!=_end; it++)
			{
				auto* mb = static_cast<irr::asset::ICPUMeshBuffer*>(*it);
				auto found = shapeCache.find(mb);
				if (found!=shapeCache.end())
					continue;
				shapeCache.insert({mb,nullptr});


				auto posAttrID = mb->getPositionAttributeIx();
				auto format = mb->getMeshDataAndFormat()->getAttribFormat(posAttrID);
				assert(format==asset::EF_R32G32B32A32_SFLOAT||format==asset::EF_R32G32B32_SFLOAT);

				auto pType = mb->getPrimitiveType();
				switch (pType)
				{
					case asset::EPT_TRIANGLE_STRIP:
						maxIndexCount = core::max((mb->getIndexCount()-2u)/3u, maxIndexCount);
						break;
					case asset::EPT_TRIANGLE_FAN:
						maxIndexCount = core::max(((mb->getIndexCount()-1u)/2u)*3u, maxIndexCount);
						break;
					case asset::EPT_TRIANGLES:
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
				makeShape(shapeCache,static_cast<irr::asset::ICPUMeshBuffer*>(*it),indices);
			delete[] indices;
		}

		template<typename Iterator>
		inline void deleteShapes(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
				rr->DeleteShape(std::get<::RadeonRays::Shape*>(*it));
		}

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
				irr::scene::IMeshSceneNode* node = *it;
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
		/*
		inline RadeonRaysIncludeLoader* getRadeonRaysGLSLIncludes()
		{
			return radeonRaysIncludes.get();
		}
		*/


		//inline auto* getRadeonRaysAPI() {return rr;}

		//
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t MaxSLI = 4u;

	protected:
		Manager(video::IVideoDriver* _driver, uint32_t _contextCount, CUcontext* _context, bool* _ownContext=nullptr);
		~Manager();
		/*
		void makeShape(MeshBufferRRShapeCache& shapeCache, const asset::ICPUMeshBuffer* mb, int32_t* indices);
		void makeInstance(	MeshNodeRRInstanceCache& instanceCache,
							const core::unordered_map<const video::IGPUMeshBuffer*,MeshBufferRRShapeCache::value_type>& GPU2CPUTable,
							scene::IMeshSceneNode* node, const int32_t* id_it);

		
		static core::smart_refctd_ptr<RadeonRaysIncludeLoader> radeonRaysIncludes;
		*/
		video::IVideoDriver* driver;
		uint32_t contextCount;
		CUcontext context[MaxSLI];
		bool ownContext[MaxSLI];
		CUstream stream[MaxSLI];

		OptixDeviceContext optixContext[MaxSLI];
		//::RadeonRays::IntersectionApi* rr;
};

}
}
}

#endif