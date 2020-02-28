// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_MESH_MANIPULATOR_H_INCLUDED__
#define __I_MESH_MANIPULATOR_H_INCLUDED__

#include <array>
#include <functional>

#include "irr/core/core.h"
#include "vector3d.h"
#include "aabbox3d.h"
#include "irr/asset/ICPUMeshBuffer.h"
#include "irr/asset/CCPUMesh.h"

namespace irr
{
namespace asset
{
	//! An interface for easy manipulation of meshes.
	/** Scale, set alpha value, flip surfaces, and so on. This exists for
	fixing problems with wrong imported or exported meshes quickly after
	loading. It is not intended for doing mesh modifications and/or
	animations during runtime.
	*/
	class IMeshManipulator : public virtual core::IReferenceCounted
	{
	public:
		//! Comparison methods
		enum E_ERROR_METRIC
		{
			/**
			Comparison with epsilon is performed by abs(original-quantized) for every significant component.
			*/
			EEM_POSITIONS,
			/**
			Comparison between vectors is based on the fact that dot product of two normalized vectors tends to 1 as the two vectors are similar.
			So that comparison is performed by dot(original, quantized)/(len(original) * len(quantized)) < (1-epsilon)
			*/
			EEM_ANGLES,
			/**
			@copydoc EEM_ANGLES
			*/
			EEM_QUATERNION,
			EEM_COUNT
		};
		//! Struct used to pass chosen comparison method and epsilon to functions performing error metrics.
		/**
		By default epsilon equals 2^-16 and EEM_POSITIONS comparison method is set.
		*/
		struct SErrorMetric
		{
			// 1.525e-5f is 2^-16
			SErrorMetric(const core::vectorSIMDf& eps = core::vectorSIMDf(1.525e-5f), E_ERROR_METRIC em = EEM_POSITIONS) : method(em), epsilon(eps) {}

			void set(E_ERROR_METRIC m, const core::vectorSIMDf& e) { method = m; epsilon = e; }

			E_ERROR_METRIC method;
			core::vectorSIMDf epsilon;
		};
		
		//vertex data needed for CSmoothNormalGenerator
		struct SSNGVertexData
		{
			uint32_t indexOffset;									//offset of the vertex into index buffer
			uint32_t hash;											//
			float wage;												//angle wage of the vertex
			core::vector4df_SIMD position;							//position of the vertex in 3D space
			core::vector3df_SIMD parentTriangleFaceNormal;			//
		};
		typedef std::function<bool(const IMeshManipulator::SSNGVertexData&, const IMeshManipulator::SSNGVertexData&, ICPUMeshBuffer*)> VxCmpFunction;

	public:
		//! Flips the direction of surfaces.
		/** Changes backfacing triangles to frontfacing
		triangles and vice versa.
		\param mesh Mesh on which the operation is performed. */
		static void flipSurfaces(ICPUMeshBuffer* inbuffer);

		//!
		static inline std::array<uint32_t,3u> getTriangleIndices(const ICPUMeshBuffer* mb, uint32_t triangleIx)
		{
			auto XXXXX = [&](auto idx) -> std::array<uint32_t,3u>
			{
				uint32_t offset;
				switch (mb->getPipeline()->getPrimitiveAssemblyParams().primitiveType)
				{
					case EPT_TRIANGLE_LIST:
						offset = triangleIx*3u;
						if (idx)
						{
							idx += offset;
							return {idx[0],idx[1],idx[2]};
						}
						else
							return {offset,offset+1u,offset+2u};
						break;
					case EPT_TRIANGLE_STRIP:
						offset = triangleIx; // 012 213
						{
							bool odd = triangleIx & 0x1u;
							auto first = odd ? 2u:1u;
							auto second = odd ? 1u:2u;
							if (idx)
							{
								idx += offset;
								return {idx[0],idx[first],idx[second]};
							}
							else
								return {offset,offset+first,offset+second};
						}
						break;
					case EPT_TRIANGLE_FAN:
						offset = triangleIx+1u;
						if (idx)
						{
							idx += offset;
							return {0u,idx[0],idx[1]};
						}
						else
							return {0u,offset,offset+1u};
						break;
					default:
						break;
				}
				assert(false);
				return {};
			};


			auto* indices = mb->getIndices();
			switch (mb->getIndexType())
			{
				case EIT_16BIT:
					return XXXXX(reinterpret_cast<const uint16_t*>(indices));
				case EIT_32BIT:
					return XXXXX(reinterpret_cast<const uint32_t*>(indices));
				default:
					return XXXXX(static_cast<const uint32_t*>(nullptr));
			}
		}

		//! Creates a copy of a mesh with all vertices unwelded
		/** \param mesh Input mesh
		\return Mesh consisting only of unique faces. All vertices
		which were previously shared are now duplicated. */
		static core::smart_refctd_ptr<ICPUMeshBuffer> createMeshBufferUniquePrimitives(ICPUMeshBuffer* inbuffer);

		//
		static core::smart_refctd_ptr<ICPUMeshBuffer> calculateSmoothNormals(ICPUMeshBuffer* inbuffer, bool makeNewMesh = false, float epsilon = 1.525e-5f,
				uint32_t normalAttrID = 3u, 
				VxCmpFunction vxcmp = [](const IMeshManipulator::SSNGVertexData& v0, const IMeshManipulator::SSNGVertexData& v1, ICPUMeshBuffer* buffer) 
				{ 
					static constexpr float cosOf45Deg = 0.70710678118f;
					return dot(v0.parentTriangleFaceNormal,v1.parentTriangleFaceNormal)[0] > cosOf45Deg;
				});

		//! Creates a copy of a mesh with vertices welded
		/** \param mesh Input mesh
        \param errMetrics Array of size EVAI_COUNT. Describes error metric for each vertex attribute (used if attribute is of floating point or normalized type).
		\param tolerance The threshold for vertex comparisons.
		\return Mesh without redundant vertices. */
		static core::smart_refctd_ptr<ICPUMeshBuffer> createMeshBufferWelded(ICPUMeshBuffer *inbuffer, const SErrorMetric* errMetrics, const bool& optimIndexType = true, const bool& makeNewMesh = false);

		//! Throws meshbuffer into full optimizing pipeline consisting of: vertices welding, z-buffer optimization, vertex cache optimization (Forsyth's algorithm), fetch optimization and attributes requantization. A new meshbuffer is created unless given meshbuffer doesn't own (getMeshDataAndFormat()==NULL) a data format descriptor.
		/**@return A new meshbuffer or NULL if an error occured. */
		static core::smart_refctd_ptr<ICPUMeshBuffer> createOptimizedMeshBuffer(const ICPUMeshBuffer* inbuffer, const SErrorMetric* _errMetric);

		//! Requantizes vertex attributes to the smallest possible types taking into account values of the attribute under consideration. A brand new vertex buffer is created and attributes are going to be interleaved in single buffer.
		/**
			The function tests type's range and precision loss after eventual requantization. The latter is performed in one of several possible methods specified
		in array parameter. By this paramater user can define method of comparison (shall depend on what the attribute's data represents) and epsilon (i.e. precision error tolerance).
		@param _meshbuffer Input meshbuffer that is to be requantized.
		@param _errMetric Array of structs defining methods of error metrics. The array must be of EVAI_COUNT length since each index of the array directly corresponds to attribute's id.
		*/
		static void requantizeMeshBuffer(ICPUMeshBuffer* _meshbuffer, const SErrorMetric* _errMetric);

		static core::smart_refctd_ptr<ICPUMeshBuffer> createMeshBufferDuplicate(const ICPUMeshBuffer* _src);

		static inline core::smart_refctd_ptr<ICPUMesh> createMeshDuplicate(const ICPUMesh* _src)
		{
			if (!_src)
				return nullptr;
	
			core::smart_refctd_ptr<ICPUMesh> dst;
			if (_src->getMeshType() == EMT_ANIMATED_SKINNED)
			{
				auto tmp = core::make_smart_refctd_ptr<CCPUSkinnedMesh>();
				//! TODO: do deep armature and keyframe copy
				tmp->setBoneReferenceHierarchy(core::smart_refctd_ptr<CFinalBoneHierarchy>(static_cast<const ICPUSkinnedMesh*>(_src)->getBoneReferenceHierarchy()));
				for (auto i=0u; i<_src->getMeshBufferCount(); i++)
					tmp->addMeshBuffer(core::smart_refctd_ptr_static_cast<ICPUSkinnedMeshBuffer>(createMeshBufferDuplicate(_src->getMeshBuffer(i))));
				dst = std::move(tmp);
			}
			else
			{
				auto tmp = core::make_smart_refctd_ptr<CCPUMesh>();
				for (auto i = 0u; i < _src->getMeshBufferCount(); i++)
					tmp->addMeshBuffer(createMeshBufferDuplicate(_src->getMeshBuffer(i)));
				dst = std::move(tmp);
			}

			return dst;
		}

        //! Creates new index buffer with invalid triangles removed.
        /**
        Invalid triangle is such consisting of two or more same indices.
        @param _input Input index buffer.
        @param _idxType Type of indices in the index buffer.
        @returns New index buffer or nullptr if input indices were of unknown type or _input was nullptr.
        */
        static void filterInvalidTriangles(ICPUMeshBuffer* _input);

        //! Creates index buffer from input converting it to indices for triangle primitives. Input is assumed to be indices for triangle strip.
        /**
        @param _input Input index buffer's data.
        @param _idxCount Index count.
        @param _idxType Type of indices (16bit or 32bit).
        */
        static core::smart_refctd_ptr<ICPUBuffer> idxBufferFromTriangleStripsToTriangles(const void* _input, size_t _idxCount, E_INDEX_TYPE _idxType);

        //! Creates index buffer from input converting it to indices for triangle primitives. Input is assumed to be indices for triangle fan.
        /**
        @param _input Input index buffer's data.
        @param _idxCount Index count.
        @param _idxType Type of indices (16bit or 32bit).
        */
        static core::smart_refctd_ptr<ICPUBuffer> idxBufferFromTrianglesFanToTriangles(const void* _input, size_t _idxCount, E_INDEX_TYPE _idxType);

        //! Compares two attributes of floating point types in accordance with passed error metric.
        /**
        @param _a First attribute.
        @param _b Second attribute.
        @param _cpa Component count.
        @param _errMetric Error metric info.
        */
        static inline bool compareFloatingPointAttribute(const core::vectorSIMDf& _a, const core::vectorSIMDf& _b, size_t _cpa, const SErrorMetric& _errMetric)
		{
			using ErrorF_t = core::vectorSIMDf(*)(core::vectorSIMDf, core::vectorSIMDf);

			ErrorF_t errorFunc = nullptr;

			switch (_errMetric.method)
			{
			case EEM_POSITIONS:
				errorFunc = [](core::vectorSIMDf _d1, core::vectorSIMDf _d2) -> core::vectorSIMDf {
					return core::abs(_d1 - _d2);
				};
				break;
			case EEM_ANGLES:
				errorFunc = [](core::vectorSIMDf _d1, core::vectorSIMDf _d2)->core::vectorSIMDf {
					_d1.w = _d2.w = 0.f;
					return core::dot(_d1, _d2) / (core::length(_d1) * core::length(_d2));
				};
				break;
			case EEM_QUATERNION:
				errorFunc = [](core::vectorSIMDf _d1, core::vectorSIMDf _d2)->core::vectorSIMDf {
					return core::dot(_d1, _d2) / (core::length(_d1) * core::length(_d2));
				};
				break;
			default:
				errorFunc = nullptr;
				break;
			}

			using CmpF_t = bool(*)(const core::vectorSIMDf&, const core::vectorSIMDf&, size_t);

			CmpF_t cmpFunc = nullptr;

			switch (_errMetric.method)
			{
			case EEM_POSITIONS:
				cmpFunc = [](const core::vectorSIMDf& _err, const core::vectorSIMDf& _epsilon, size_t _cpa) -> bool {
					for (size_t i = 0u; i < _cpa; ++i)
						if (_err.pointer[i] > _epsilon.pointer[i])
							return false;
					return true;
				};
				break;
			case EEM_ANGLES:
			case EEM_QUATERNION:
				cmpFunc = [](const core::vectorSIMDf& _err, const core::vectorSIMDf& _epsilon, size_t _cpa) -> bool {
					return _err.x > (1.f - _epsilon.x);
				};
				break;
			default:
				cmpFunc = nullptr;
				break;
			}

			_IRR_DEBUG_BREAK_IF(!errorFunc)
				_IRR_DEBUG_BREAK_IF(!cmpFunc)
				if (!errorFunc || !cmpFunc)
					return false;

			const core::vectorSIMDf err = errorFunc(_a, _b);
			return cmpFunc(err, _errMetric.epsilon, _cpa);
		}

		//! Get amount of polygons in mesh buffer.
		/** \param meshbuffer Input mesh buffer
		\param Outputted Number of polygons in mesh buffer, if successful.
		\return If successfully can provide information */
        template<typename ...MeshbufTemplParams>
		static inline bool getPolyCount(uint32_t& outCount, IMeshBuffer<MeshbufTemplParams...>* meshbuffer)
		{
			outCount = 0;
			if (!meshbuffer)
				return false;
            if (!meshbuffer->getPipeline())
                return false;

            const E_PRIMITIVE_TOPOLOGY primType = meshbuffer->getPipeline()->getPrimitiveAssemblyParams().primitiveType;

			uint32_t trianglecount;

			switch (primType)
			{
			case EPT_POINT_LIST:
				trianglecount = meshbuffer->getIndexCount();
				break;
			case EPT_LINE_STRIP:
				trianglecount = meshbuffer->getIndexCount() - 1;
				break;
			case EPT_LINE_LIST:
				trianglecount = meshbuffer->getIndexCount() / 2;
				break;
			case EPT_TRIANGLE_STRIP:
				trianglecount = meshbuffer->getIndexCount() - 2;
				break;
			case EPT_TRIANGLE_FAN:
				trianglecount = meshbuffer->getIndexCount() - 2;
				break;
			case EPT_TRIANGLE_LIST:
				trianglecount = meshbuffer->getIndexCount() / 3;
				break;
			}

			outCount = trianglecount;
			return true;
		}

		//! Get amount of polygons in mesh.
		/** \param meshbuffer Input mesh
		\param Outputted Number of polygons in mesh, if successful.
		\return If successfully can provide information */
		template<typename T>
		static inline bool getPolyCount(uint32_t& outCount, IMesh<T>* mesh)
		{
			outCount = 0u;
			if (!mesh)
				return false;

			bool retval = true;
			for (uint32_t g = 0; g < mesh->getMeshBufferCount(); ++g)
			{
				uint32_t trianglecount;
				retval = retval && getPolyCount(trianglecount, mesh->getMeshBuffer(g));
                outCount += trianglecount;
			}

			return retval;
		}
    protected:
};

} // end namespace scene
} // end namespace irr


#endif
