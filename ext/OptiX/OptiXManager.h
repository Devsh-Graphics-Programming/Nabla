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

		using RegisteredBufferCache = core::set<cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>>;
		template<typename Iterator>
		OptixTraversableHandle createAccelerationStructure(CUstream stream, RegisteredBufferCache& bufferCache, const OptixAccelBuildOptions& accelOptions, Iterator _begin, Iterator _end, uint32_t deviceID=0u, size_t scratchBufferSize=0u, CUdeviceptr scratchBuffer = nullptr)
		{
			auto EFormatToOptixFormat = [](asset::E_FORMAT format) -> OptixVertexFormat
			{
				switch (format)
				{
				case asset::EF_R16G16_SNORM:
					return OPTIX_VERTEX_FORMAT_SNORM16_2;
				case asset::EF_R32G32B32_SNORM:
					return OPTIX_VERTEX_FORMAT_SNORM16_3;
				case asset::EF_R16G16_SFLOAT:
					return OPTIX_VERTEX_FORMAT_HALF2;
				case asset::EF_R16G16B16_SFLOAT:
					return OPTIX_VERTEX_FORMAT_HALF3;
				case asset::EF_R32G32_SFLOAT:
					return OPTIX_VERTEX_FORMAT_FLOAT2;
				case asset::EF_R32G32B32_SFLOAT:
					return OPTIX_VERTEX_FORMAT_FLOAT3;
				default:
					break;
				}
				return static_cast<OptixVertexFormat>(0u);
			};

			const auto inputCount = std::distance(_begin,_end);
			core::vector<CUgraphicsResource> vertexBuffers;
			core::vector<uint32_t> bufferIndices(inputCount);
			{
				vertexBuffers.reserve(inputCount);

				RegisteredBufferCache newBuffers;
				uint32_t counter = 0u;
				for (auto it=_begin; it!=_end; it++,counter++)
				{
					bufferIndices[counter] = ~0u;

					auto* mb = static_cast<video::IGPUMeshBuffer*>(*it);
					auto pipeline = mb->getMeshDataAndFormat();
					auto posIx = mb->getPositionAttributeIx();
					if (EFormatToOptixFormat(pipeline->getAttribFormat(posIx)))
						continue;

					auto posbuffer = pipeline->getMappedBuffer(posIx);
					auto found = bufferCache.find(posbuffer);
					if (found == bufferCache.end())
					{
						RegisteredBufferCache::value_type link;
						if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&link,CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)))
							continue;
						found = newBuffers.insert(std::move(link)).first;
					}
					vertexBuffers.push_back(*found);
					bufferIndices[counter] = counter;
				}
				bufferCache.merge(newBuffers);
			}

			if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuGraphicsMapResources(vertexBuffers.size(),vertexBuffers.data(),stream)))
				return nullptr;

			constexpr uint32_t buildStepSize = 256u;
			const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE }; // TODO: MAYBE CHANGE?
			auto it = _begin;
			for (auto i=0; i<inputCount;)
			{
				OptixBuildInput buildInputs[buildStepSize] = {};
				OptixAccelBufferSizes buffSizes = {};
				uint32_t j = 0u;
				for (; j<buildStepSize&&it!=_end; it++,j++,i++)
				{
					auto buffIx = bufferIndices[i];
					if (buffIx == (~0u))
						continue;

					auto* mb = static_cast<video::IGPUMeshBuffer*>(*it);
					auto pipeline = mb->getMeshDataAndFormat();
					auto posIx = mb->getPositionAttributeIx();

					buildInputs[j].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
					buildInputs[j].triangleArray.vertexBuffers = vertexBuffers.data()+buffIx;
					//buildInputs[j].triangleArray.numVertices = mb-> ? ;
					buildInputs[j].triangleArray.vertexFormat = EFormatToOptixFormat(pipeline->getAttribFormat(posIx));
					buildInputs[j].triangleArray.vertexStrideInBytes = pipeline->getAttribStride(posIx);
					if (pipeline->getIndexBuffer())
					{
						//buildInputs[j].triangleArray.indexBuffer = ;
						buildInputs[j].triangleArray.numIndexTriplets = mb->getIndexCount()/3u;
						buildInputs[j].triangleArray.indexFormat = mb->getIndexType()!=asset::EIT_16BIT ? OPTIX_INDICES_FORMAT_UNSIGNED_INT3:OPTIX_INDICES_FORMAT_UNSIGNEDSHORT3;
						buildInputs[j].triangleArray.indexStrideInBytes = 0u;
						buildInputs[j].triangleArray.primitiveIndexOffset = mb->getBaseVertex();
						buildInputs[j].triangleArray.numSbtRecords = 1u;
						buildInputs[j].triangleArray.flags = triangle_input_flags;
					}
				}
				optixAccelComputeMemoryUsage(optixContext[deviceID],&accelOptions,buildInputs,j,&buffSizes);

				// check and resize buffers as necessary
				buffSizes.outputSizeInBytes
				buffSizes.tempSzeInBytes
				optixAccelBuild(optixContext[deviceID],stream,&accelOptions,buildInputs,j,scratchBuffer,scratchBuffer->getSize(),outputBuffer,outputBuffer->getSize(),outputHandle,props,propsCount);
			}

			if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuGraphicsUnmapResources(vertexBuffers.size(),vertexBuffers.data(), stream)))
				return nullptr;

			return nullptr;
		}

		// TODO: optixAccelCompact

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