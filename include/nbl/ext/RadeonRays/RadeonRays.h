// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_EXT_RADEON_RAYS_H_INCLUDED__
#define __NBL_EXT_RADEON_RAYS_H_INCLUDED__

#include "nabla.h"

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

// this is a really bad mock (do not take inspiration from it for the real API)
class NBL_API MockSceneManager
{
	public:
		using MeshBufferGUID = uint32_t;
		using ObjectGUID = uint32_t;

		struct ObjectData
		{
			core::matrix3x4SIMD tform;
			core::smart_refctd_ptr<video::IGPUMesh> mesh;
			core::vector<MeshBufferGUID> instanceGUIDPerMeshBuffer;
		};

		const ObjectData& getObjectData(const ObjectGUID guid) const {return m_objectData[guid];}

		//mocks
		core::vector<ObjectData> m_objectData;
};


class NBL_API Manager final : public core::IReferenceCounted
{
	public:
		static core::smart_refctd_ptr<Manager> create(video::IVideoDriver* _driver);

		
		std::pair<::RadeonRays::Buffer*,cl_mem> linkBuffer(const video::IGPUBuffer* buffer, cl_mem_flags access);
		void unlinkBuffer(std::pair<::RadeonRays::Buffer*,cl_mem>&& link);


		struct MeshBufferRRShapeCache
		{
			core::unordered_map<const asset::ICPUMeshBuffer*,::RadeonRays::Shape*> m_cpuAssociative;
			core::unordered_map<const video::IGPUMeshBuffer*,::RadeonRays::Shape*> m_gpuAssociative;
		};
		using NblInstanceRRInstanceCache = core::unordered_map<MockSceneManager::ObjectGUID,core::smart_refctd_dynamic_array<::RadeonRays::Shape*>>;

		template<typename Iterator>
		inline void makeRRShapes(MeshBufferRRShapeCache& shapeCache, Iterator _begin, Iterator _end)
		{
			auto& cpuCache = shapeCache.m_cpuAssociative;
			cpuCache.reserve(cpuCache.size()+std::distance(_begin,_end));
			shapeCache.m_gpuAssociative.reserve(cpuCache.size());

			for (auto it=_begin; it!=_end; it++)
			{
				auto* mb = static_cast<nbl::asset::ICPUMeshBuffer*>(*it);
				auto found = cpuCache.find(mb);
				if (found!=cpuCache.end())
					continue;
				cpuCache.insert({mb,nullptr});


				const auto posAttrID = mb->getPositionAttributeIx();
				const auto format = mb->getAttribFormat(posAttrID);
				assert(format==asset::EF_R32G32B32A32_SFLOAT||format==asset::EF_R32G32B32_SFLOAT);

				assert(mb->getPipeline()->getPrimitiveAssemblyParams().primitiveType==EPT_TRIANGLE_LIST);
				assert(mb->getIndexBufferBinding().buffer);
			}

			for (auto it=_begin; it!=_end; it++)
				makeShape(cpuCache,static_cast<nbl::asset::ICPUMeshBuffer*>(*it));
		}

		template<typename Iterator>
		inline void deleteShapes(Iterator _begin, Iterator _end)
		{
			for (auto it = _begin; it != _end; it++)
			{
				rr->DeleteShape(std::get<::RadeonRays::Shape*>(*it));
			}
		}


		template<typename Iterator>
		inline void makeRRInstances(NblInstanceRRInstanceCache& instanceCache, MockSceneManager* mock_smgr,
									MeshBufferRRShapeCache& shapeCache, asset::IAssetManager* _assetManager,
									Iterator _objectsBegin, Iterator _objectsEnd)
		{
			for (auto it=_objectsBegin; it!=_objectsEnd; it++)
			{
				const MockSceneManager::ObjectGUID objectGUID = *it;
				makeInstance(instanceCache,mock_smgr,shapeCache,_assetManager,objectGUID);
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

		static inline void shapeSetTransform(::RadeonRays::Shape* shape, const core::matrix3x4SIMD& transform)
		{
			core::matrix4SIMD tform(transform);

			// TODO: move this stuff
			struct dvec3
			{
				dvec3() : x(0.0), y(0.0), z(0.0) {}
				dvec3(const core::vectorSIMDf& p) : x(p.x), y(p.y), z(p.z) {}

				dvec3& operator-=(dvec3 other)
				{
					x -= other.x;
					y -= other.y;
					z -= other.z;
					return *this;
				}
				dvec3& operator*=(double other)
				{
					x *= other;
					y *= other;
					z *= other;
					return *this;
				}
				dvec3 operator*(double other) const
				{
					dvec3 retval(*this);
					retval *= other;
					return retval;
				}
				dvec3& operator/=(double other)
				{
					x /= other;
					y /= other;
					z /= other;
					return *this;
				}

				double x,y,z;
			};
			auto dot = [](const dvec3& a, const dvec3& b) -> double
			{
				return a.x*b.x+a.y*b.y+a.z*b.z;
			};
			auto cross = [](const dvec3& a, const dvec3& b) -> dvec3
			{
				dvec3 retval;
				retval.x = a.y*b.z-a.z*b.y;
				retval.y = a.z*b.x-a.x*b.z;
				retval.z = a.x*b.y-a.y*b.x;
				return retval;
			};
			const dvec3 in_rows[3] = {transform.rows[0],transform.rows[1],transform.rows[2]};
			dvec3 out_cols[4];
			out_cols[0] = cross(in_rows[1],in_rows[2]);
			out_cols[1] = cross(in_rows[2],in_rows[0]);
			out_cols[2] = cross(in_rows[0],in_rows[1]);
			const double determinant = dot(in_rows[0],out_cols[0]);
			//if (core::isnan(determinant)||determinant<FLT_MIN)
				//exit();
			out_cols[3] = {};
			const auto translation = transform.getTranslation();
			for (auto i=0; i<3; i++)
			{
				out_cols[3] -= out_cols[i]*double(translation.pointer[i]);
				out_cols[i] /= determinant;
			}
			out_cols[3] /= determinant;

			core::matrix4SIMD tform_inv;
			for (auto i=0; i<4; i++)
			{
				tform_inv.rows[i].x = out_cols[i].x;
				tform_inv.rows[i].y = out_cols[i].y;
				tform_inv.rows[i].z = out_cols[i].z;
			}
			tform_inv = core::transpose(tform_inv);

			shape->SetTransform(reinterpret_cast<::RadeonRays::matrix&>(tform),reinterpret_cast<::RadeonRays::matrix&>(tform_inv));
		}
		template<typename Iterator>
		inline void update(const MockSceneManager* mock_smgr, Iterator _instancesBegin, Iterator _instancesEnd)
		{
			bool needToCommit = false;
			for (auto it=_instancesBegin; it!=_instancesEnd; it++)
			{
				const MockSceneManager::ObjectGUID objectID = it->first;
				const auto* shapeArray = it->second.get();

				const auto firstShape = shapeArray->operator[](0);

				// TODO: when actually implemented smgr, need a way to pull absolute transforms from GPU or CPU for RR to use
				const auto& absoluteTForm = mock_smgr->getObjectData(objectID).tform;
				// check if moved
				core::matrix4SIMD oldTForm, dummy;
				firstShape->GetTransform(reinterpret_cast<::RadeonRays::matrix&>(oldTForm), reinterpret_cast<::RadeonRays::matrix&>(dummy));
				if (!core::equals(absoluteTForm,oldTForm.extractSub3x4(),core::matrix3x4SIMD(0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f)))
				{
					for (auto shape : *shapeArray)
						shapeSetTransform(shape,absoluteTForm);
					needToCommit = true;
				}
			}

			if (needToCommit)
				rr->Commit();
		}

		inline bool hasImplicitCL2GLSync() const { return m_automaticOpenCLSync; }


		inline auto* getRadeonRaysAPI() {return rr;}

		inline cl_command_queue getCLCommandQueue() { return commandQueue; }

	protected:
		Manager(video::IVideoDriver* _driver, cl_context context, bool automaticOpenCLSync);
		~Manager();

		void makeShape(core::unordered_map<const asset::ICPUMeshBuffer*,::RadeonRays::Shape*>& cpuCache, const asset::ICPUMeshBuffer* mb);
		void makeInstance(	NblInstanceRRInstanceCache& instanceCache, MockSceneManager* mock_smgr,
							MeshBufferRRShapeCache& shapeCache, asset::IAssetManager* _assetManager,
							MockSceneManager::ObjectGUID objectID);

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