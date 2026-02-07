// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_POLYGON_GEOMETRY_MANIPULATOR_H_INCLUDED_
#define _NBL_ASSET_C_POLYGON_GEOMETRY_MANIPULATOR_H_INCLUDED_


#include "nbl/core/declarations.h"
#include "nbl/core/hash/blake.h"

#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/utils/CGeometryManipulator.h"
#include "nbl/asset/utils/CSmoothNormalGenerator.h"
#include "nbl/asset/utils/COBBGenerator.h"
#include "nbl/builtin/hlsl/shapes/obb.hlsl"

namespace nbl::asset
{

//! An interface for easy manipulation of polygon geometries.
class NBL_API2 CPolygonGeometryManipulator
{
	public:
		static core::blake3_hash_t computeDeterministicContentHash(const ICPUPolygonGeometry* geo);

		static inline void recomputeContentHashes(ICPUPolygonGeometry* geo)
		{
			if (!geo)
				return;
			CGeometryManipulator::recomputeContentHash(geo->getPositionView());
			CGeometryManipulator::recomputeContentHash(geo->getIndexView());
			CGeometryManipulator::recomputeContentHash(geo->getNormalView());
			for (const auto& view : *geo->getJointWeightViews())
			{
				CGeometryManipulator::recomputeContentHash(view.indices);
				CGeometryManipulator::recomputeContentHash(view.weights);
			}
			if (auto pView=geo->getJointOBBView(); pView)
				CGeometryManipulator::recomputeContentHash(*pView);
			for (const auto& view : *geo->getAuxAttributeViews())
				CGeometryManipulator::recomputeContentHash(view);
		}

		//
		static inline void recomputeRanges(ICPUPolygonGeometry* geo, const bool deduceRangeFormats=true)
		{
			if (!geo)
				return;
			auto recomputeRange = [deduceRangeFormats](const IGeometry<ICPUBuffer>::SDataView& view)->void
			{
				CGeometryManipulator::recomputeRange(const_cast<IGeometry<ICPUBuffer>::SDataView&>(view),deduceRangeFormats);
			};
			recomputeRange(geo->getPositionView());
			recomputeRange(geo->getIndexView());
			recomputeRange(geo->getNormalView());
			for (const auto& view : *geo->getJointWeightViews())
			{
				recomputeRange(view.indices);
				recomputeRange(view.weights);
			}
			if (auto pView=geo->getJointOBBView(); pView)
				recomputeRange(*pView);
			for (const auto& view : *geo->getAuxAttributeViews())
				recomputeRange(view);
		}

		//
		static inline IGeometryBase::SAABBStorage computeAABB(const ICPUPolygonGeometry* geo)
		{
			if (!geo || !geo->getPositionView() || geo->getPositionView().composed.rangeFormat>=IGeometryBase::EAABBFormat::Count)
				return {};

			if (geo->getIndexView() || geo->isSkinned())
			{
				const auto jointViewCount = geo->getJointWeightViews().size();
				auto isVertexSkinned = [&jointViewCount](const ICPUPolygonGeometry* geo, uint64_t vertex_i)
				{
					if (!geo->isSkinned()) return false;
					for (auto weightView_i = 0u; weightView_i < jointViewCount; weightView_i++)
					{
						const auto& weightView = geo->getJointWeightViews()[weightView_i];
						hlsl::float32_t4 weight;
						weightView.weights.decodeElement(vertex_i, weight);
						for (auto channel_i = 0; channel_i < getFormatChannelCount(weightView.weights.composed.format); channel_i++)
						if (weight[channel_i] > 0.f)
							return true;
					}
					return false;
				};

				auto addToAABB = [&](auto& aabb)->void
				{
					using aabb_t = std::remove_reference_t<decltype(aabb)>;
					if (geo->getIndexView())
					{
						for (auto index_i = 0u; index_i != geo->getIndexView().getElementCount(); index_i++)
						{
							hlsl::vector<uint32_t, 1> vertex_i;
							geo->getIndexView().decodeElement(index_i, vertex_i);
							if (isVertexSkinned(geo, vertex_i.x)) continue;
							typename aabb_t::point_t pt;
							geo->getPositionView().decodeElement(vertex_i.x, pt);
							aabb.addPoint(pt);
						}
					} else
					{
						for (auto vertex_i = 0u; vertex_i != geo->getPositionView().getElementCount(); vertex_i++)
						{
							if (isVertexSkinned(geo, vertex_i)) continue;
							typename aabb_t::point_t pt;
							geo->getPositionView().decodeElement(vertex_i, pt);
							aabb.addPoint(pt);
						}
					}
				};
				IGeometryBase::SDataViewBase tmp = geo->getPositionView().composed;
				tmp.resetRange();
				tmp.visitRange(addToAABB);
				return tmp.encodedDataRange;
			}
			else
			{
				return geo->getPositionView().composed.encodedDataRange;
			}
		}

		static inline void recomputeAABB(const ICPUPolygonGeometry* geo)
		{
			if (geo->isMutable())
				const_cast<IGeometryBase::SAABBStorage&>(geo->getAABBStorage()) = computeAABB(geo);
		}

		static inline core::smart_refctd_ptr<ICPUPolygonGeometry> createTriangleListIndexing(const ICPUPolygonGeometry* geo)
		{
			const auto* indexing = geo->getIndexingCallback();
			if (!indexing) return nullptr;
			if (indexing->degree() != 3) return nullptr;

			const auto originalView = geo->getIndexView();
			const auto originalIndexSize = originalView ? originalView.composed.stride : 0;
			const auto primCount = geo->getPrimitiveCount();
			const auto maxIndex = geo->getPositionView().getElementCount() - 1;
			const uint8_t indexSize = maxIndex <= std::numeric_limits<uint16_t>::max() ? sizeof(uint16_t) : sizeof(uint32_t);
			const auto outGeometry = core::move_and_static_cast<ICPUPolygonGeometry>(geo->clone(0u));

			if (indexing && indexing->knownTopology() == EPT_TRIANGLE_LIST) 
				return outGeometry;

			
			auto* outGeo = outGeometry.get();
			const auto indexBufferUsages = [&]
			{
					if (originalView) return originalView.src.buffer->getUsageFlags();
					return core::bitflag<IBuffer::E_USAGE_FLAGS>(IBuffer::EUF_INDEX_BUFFER_BIT);
			}();
			auto indexBuffer = ICPUBuffer::create({ primCount * indexing->degree() * indexSize, indexBufferUsages });
			auto indexBufferPtr = indexBuffer->getPointer();
			auto indexView = ICPUPolygonGeometry::SDataView{
				.composed = {
					.stride = indexSize,
				},
				.src = {
					.offset = 0,
					.size = indexBuffer->getSize(),
					.buffer = std::move(indexBuffer)
				}
			};

			switch (indexSize)
			{
				case 2:
				{
					IPolygonGeometryBase::IIndexingCallback::SContext<uint16_t> context{
						.indexBuffer = geo->getIndexView().getPointer(),
						.indexSize = originalIndexSize,
						.beginPrimitive = 0,
						.endPrimitive = primCount,
						.out = indexBufferPtr,
					};
					indexing->operator()(context);

					indexView.composed.encodedDataRange.u16.minVx[0] = 0;
					indexView.composed.encodedDataRange.u16.maxVx[0] = maxIndex;
					indexView.composed.format = EF_R16_UINT;
					indexView.composed.rangeFormat = IGeometryBase::EAABBFormat::U16;
					break;
				}
				case 4:
				{
					IPolygonGeometryBase::IIndexingCallback::SContext<uint32_t> context{
						.indexBuffer = geo->getIndexView().getPointer(),
						.indexSize = originalIndexSize,
						.beginPrimitive = 0,
						.endPrimitive = primCount,
						.out = indexBufferPtr,
					};
					indexing->operator()(context);

					indexView.composed.encodedDataRange.u32.minVx[0] = 0;
					indexView.composed.encodedDataRange.u32.maxVx[0] = maxIndex;
					indexView.composed.format = EF_R32_UINT;
					indexView.composed.rangeFormat = IGeometryBase::EAABBFormat::U32;
					break;
				}
				default:
				{
					assert(false);
					return nullptr;
				}
			}
			 
			outGeo->setIndexing(IPolygonGeometryBase::TriangleList());
			outGeo->setIndexView(std::move(indexView));
			CGeometryManipulator::recomputeContentHash(outGeo->getIndexView());

			return outGeometry;
		}

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

    template <typename FetchVertexFn> 
      requires (std::same_as<std::invoke_result_t<FetchVertexFn, size_t>, hlsl::float32_t3>)
    static inline hlsl::shapes::OBB<3, hlsl::float32_t> calculateOBB(size_t vertexCount, FetchVertexFn&& fetchFn, float epsilon = 1.525e-5f)
    {
			return COBBGenerator::compute(vertexCount, std::forward<FetchVertexFn>(fetchFn), epsilon);
    }

		static core::smart_refctd_ptr<ICPUPolygonGeometry> createUnweldedList(const ICPUPolygonGeometry* inGeo);

		using SSNGVertexData = CSmoothNormalGenerator::VertexData;
		using SSNGVxCmpFunction = CSmoothNormalGenerator::VxCmpFunction;

		static core::smart_refctd_ptr<ICPUPolygonGeometry> createSmoothVertexNormal(const ICPUPolygonGeometry* inbuffer, bool enableWelding = false, float epsilon = 1.525e-5f,
				SSNGVxCmpFunction vxcmp = [](const SSNGVertexData& v0, const SSNGVertexData& v1, const ICPUPolygonGeometry* buffer) 
				{ 
					constexpr float cosOf45Deg = 0.70710678118f;
					return dot(normalize(v0.weightedNormal),normalize(v1.weightedNormal)) > cosOf45Deg;
				});

#if 0 // TODO: REDO
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
						if ((_d1==core::vectorSIMDf(0.f)).all() || (_d2==core::vectorSIMDf(0.f)).all())
							return core::vectorSIMDf(-INFINITY);
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

			_NBL_DEBUG_BREAK_IF(!errorFunc)
				_NBL_DEBUG_BREAK_IF(!cmpFunc)
				if (!errorFunc || !cmpFunc)
					return false;

			const core::vectorSIMDf err = errorFunc(_a, _b);
			return cmpFunc(err, _errMetric.epsilon, _cpa);
		}

		//!
		static inline uint32_t calcVertexSize(const ICPUMeshBuffer* meshbuffer)
		{
			const auto* ppln = meshbuffer->getPipeline();
			if (!ppln)
				return 0u;

			const auto& vtxInputParams = ppln->getCachedCreationParams().vertexInput;
			uint32_t size = 0u;
			for (uint32_t i=0u; i<ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT; ++i)
			if (vtxInputParams.enabledAttribFlags & (1u<<i))
				size += asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(vtxInputParams.attributes[i].format));
			return size;
		}
		

				//! Swaps the index buffer for a new index buffer with invalid triangles removed.
				/**
				Invalid triangle is such consisting of two or more same indices.
				@param _input Input index buffer.
				@param _idxType Type of indices in the index buffer.
				@returns New index buffer or nullptr if input indices were of unknown type or _input was nullptr.
				*/
				static void filterInvalidTriangles(ICPUMeshBuffer* _input);

				//! Creates index buffer from input converting it to indices for line list primitives. Input is assumed to be indices for line strip.
				/**
				@param _input Input index buffer's data.
				@param _idxCount Index count.
				@param _inIndexType Type of input index buffer data (32bit or 16bit).
				@param _outIndexType Type of output index buffer data (32bit or 16bit).
				*/
				static core::smart_refctd_ptr<ICPUBuffer> idxBufferFromLineStripsToLines(const void* _input, uint32_t& _idxCount, E_INDEX_TYPE _inIndexType, E_INDEX_TYPE _outIndexType);

				//! Creates index buffer from input converting it to indices for triangle list primitives. Input is assumed to be indices for triangle strip.
				/**
				@param _input Input index buffer's data.
				@param _idxCount Index count.
				@param _inIndexType Type of input index buffer data (32bit or 16bit).
				@param _outIndexType Type of output index buffer data (32bit or 16bit).
				*/
				static core::smart_refctd_ptr<ICPUBuffer> idxBufferFromTriangleStripsToTriangles(const void* _input, uint32_t& _idxCount, E_INDEX_TYPE _inIndexType, E_INDEX_TYPE _outIndexType);

				//! Creates index buffer from input converting it to indices for triangle list primitives. Input is assumed to be indices for triangle fan.
				/**
				@param _input Input index buffer's data.
				@param _idxCount Index count.
				@param _inIndexType Type of input index buffer data (32bit or 16bit).
				@param _outIndexType Type of output index buffer data (32bit or 16bit).
				*/
				static core::smart_refctd_ptr<ICPUBuffer> idxBufferFromTrianglesFanToTriangles(const void* _input, uint32_t& _idxCount, E_INDEX_TYPE _inIndexType, E_INDEX_TYPE _outIndexType);

		//!
		static inline std::array<uint32_t,3u> getTriangleIndices(const ICPUMeshBuffer* mb, uint32_t triangleIx)
		{
			auto XXXXX = [&](auto idx) -> std::array<uint32_t,3u>
			{
				uint32_t offset;
				switch (mb->getPipeline()->getCachedCreationParams().primitiveAssembly.primitiveType)
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

		//!
		static inline uint32_t upperBoundVertexID(const ICPUMeshBuffer* meshbuffer)
		{
			uint32_t vertexCount = 0u;
			auto impl = [meshbuffer,&vertexCount](const auto* indexPtr) -> void
			{
				const uint32_t indexCount = meshbuffer->getIndexCount();
				if (indexPtr)
				{
					for (uint32_t j=0ull; j<indexCount; j++)
					{
						uint32_t index = static_cast<uint32_t>(indexPtr[j]);
						if (index>vertexCount)
							vertexCount = index;
					}
					if (indexCount)
						vertexCount++;
				}
				else
					vertexCount = indexCount;
			};

			const void* indices = meshbuffer->getIndices();
			switch (meshbuffer->getIndexType())
			{
				case EIT_32BIT:
					impl(reinterpret_cast<const uint32_t*>(indices));
					break;
				case EIT_16BIT:
					impl(reinterpret_cast<const uint16_t*>(indices));
					break;
				default:
					vertexCount = meshbuffer->getIndexCount();
					break;
			}

			return vertexCount;
		}

		static float DistanceToLine(core::vectorSIMDf P0, core::vectorSIMDf P1, core::vectorSIMDf InPoint);
		static float DistanceToPlane(core::vectorSIMDf InPoint, core::vectorSIMDf PlanePoint, core::vectorSIMDf PlaneNormal);

		//! Calculates bounding box of the meshbuffer
		static inline core::aabbox3df calculateBoundingBox(
			const ICPUMeshBuffer* meshbuffer, core::aabbox3df* outJointAABBs=nullptr,
			uint32_t indexCountOverride=0u, const void* indexBufferOverride=nullptr,
			E_INDEX_TYPE indexTypeOverride=static_cast<E_INDEX_TYPE>(~0u)
		)
		{
			core::aabbox3df aabb(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX);
			if (!meshbuffer->getPipeline())
				return aabb;
			
			auto posAttrId = meshbuffer->getPositionAttributeIx();
			const ICPUBuffer* mappedAttrBuf = meshbuffer->getAttribBoundBuffer(posAttrId).buffer.get();
			if (posAttrId >= ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT || !mappedAttrBuf)
				return aabb;
			
			const bool computeJointAABBs = outJointAABBs&&meshbuffer->isSkinned();
			if (computeJointAABBs)
			for (auto i=0u; i<meshbuffer->getJointCount(); i++)
				outJointAABBs[i] = aabb;

			if (indexCountOverride==0u)
				indexCountOverride = meshbuffer->getIndexCount();
			auto impl = [meshbuffer,computeJointAABBs,&aabb,indexCountOverride](const auto* indexPtr, auto* jointAABBs) -> void
			{
				const uint32_t jointCount = meshbuffer->getJointCount();
				const auto jointIDAttr = meshbuffer->getJointIDAttributeIx();
				const auto jointWeightAttrId = meshbuffer->getJointWeightAttributeIx();
				const auto maxInfluences = core::min(meshbuffer->deduceMaxJointsPerVertex(), meshbuffer->getMaxJointsPerVertex());
				const uint32_t maxWeights = computeJointAABBs ? getFormatChannelCount(meshbuffer->getAttribFormat(jointWeightAttrId)):0u;
				const auto* inverseBindPoses = meshbuffer->getInverseBindPoses();

				for (uint32_t j=0u; j<indexCountOverride; j++)
				{
					uint32_t ix;
					if constexpr (std::is_void_v<std::remove_pointer_t<decltype(indexPtr)>>)
						ix = j;
					else
						ix = indexPtr[j];
					const auto pos = meshbuffer->getPosition(ix);

					bool noJointInfluence = true;
					if constexpr (!std::is_void_v<std::remove_pointer_t<decltype(jointAABBs)>>)
					{
						uint32_t jointIDs[4u];
						meshbuffer->getAttribute(jointIDs,jointIDAttr,ix);
						core::vectorSIMDf weights;
						meshbuffer->getAttribute(weights,jointWeightAttrId,ix);
						float weightRemainder = 1.f;
						for (auto i=0u; i<maxInfluences; i++)
						{
							const auto jointID = jointIDs[i];
							if (jointID<jointCount)
							if ((i<maxWeights ? weights[i]:weightRemainder)>FLT_MIN)
							{
								core::vectorSIMDf boneSpacePos;
								inverseBindPoses[jointID].transformVect(boneSpacePos,pos);
								jointAABBs[jointID].addInternalPoint(boneSpacePos.getAsVector3df());
								noJointInfluence = false;
							}
							weightRemainder -= weights[i];
						}
					}
					
					if (noJointInfluence)
						aabb.addInternalPoint(pos.getAsVector3df());
				}
			};

			if (!indexBufferOverride)
				indexBufferOverride = meshbuffer->getIndices();
			if (indexTypeOverride>EIT_UNKNOWN)
				indexTypeOverride = meshbuffer->getIndexType();
			void* void_null = nullptr;
			switch (indexTypeOverride)
			{
				case EIT_32BIT:
					if (computeJointAABBs)
						impl(reinterpret_cast<const uint32_t*>(indexBufferOverride),outJointAABBs);
					else
						impl(reinterpret_cast<const uint32_t*>(indexBufferOverride),void_null);
					break;
				case EIT_16BIT:
					if (computeJointAABBs)
						impl(reinterpret_cast<const uint16_t*>(indexBufferOverride),outJointAABBs);
					else
						impl(reinterpret_cast<const uint16_t*>(indexBufferOverride),void_null);
					break;
				default:
					if (computeJointAABBs)
						impl(void_null,outJointAABBs);
					else
						impl(void_null,void_null);
					break;
			}
			return aabb;
		}

		//! Recalculates the cached bounding box of the meshbuffer
		static inline void recalculateBoundingBox(ICPUMeshBuffer* meshbuffer)
		{
			meshbuffer->setBoundingBox(calculateBoundingBox(meshbuffer,meshbuffer->getJointAABBs()));
		}

		//! Flips the direction of surfaces.
		/** Changes backfacing triangles to frontfacing
		triangles and vice versa.
		\param mesh Mesh on which the operation is performed. */
		static void flipSurfaces(ICPUMeshBuffer* inbuffer);

		//! Creates a copy of a mesh with all vertices unwelded
		/** \param mesh Input mesh
		\return Mesh consisting only of unique faces. All vertices
		which were previously shared are now duplicated. */
		static core::smart_refctd_ptr<ICPUMeshBuffer> createMeshBufferUniquePrimitives(ICPUMeshBuffer* inbuffer, bool _makeIndexBuf = false);

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

				//! Creates a 32bit index buffer for a mesh with primitive types changed to list types
				/**#
		@param _newPrimitiveType
				@param _begin non-const iterator to beginning of meshbuffer range
				@param _end non-const iterator to ending of meshbuffer range
				*/
		template<typename Iterator>
		static inline void homogenizePrimitiveTypeAndIndices(Iterator _begin, Iterator _end, const E_PRIMITIVE_TOPOLOGY _newPrimitiveType, const E_INDEX_TYPE outIndexType = EIT_32BIT)
		{
			// analyse
			uint32_t iotaLength = 0u;
			uint32_t patchVertexCount = 0u;
			for (auto it=_begin; it!=_end; it++)
			{
				auto& cpumb = *it;
				assert(cpumb->isMutable());
				//assert(!IPreHashed::anyDependantDiscardedContents(cpumb));

				const auto& params = cpumb->getPipeline()->getCachedCreationParams().primitiveAssembly;
				switch (params.primitiveType)
				{
					case EPT_POINT_LIST:
						assert(_newPrimitiveType==EPT_POINT_LIST);
						break;
					case EPT_LINE_LIST:
						assert(_newPrimitiveType==EPT_LINE_LIST);
						break;
					case EPT_LINE_STRIP:
						assert(_newPrimitiveType==EPT_LINE_LIST);
						break;
					case EPT_TRIANGLE_LIST:
						assert(_newPrimitiveType==EPT_TRIANGLE_LIST);
						break;
					case EPT_TRIANGLE_STRIP:
						assert(_newPrimitiveType==EPT_TRIANGLE_LIST);
						break;
					case EPT_TRIANGLE_FAN:
						assert(_newPrimitiveType==EPT_TRIANGLE_LIST);
						break;
					case EPT_PATCH_LIST:
						assert(_newPrimitiveType==EPT_PATCH_LIST);
						if (patchVertexCount)
							assert(params.tessPatchVertCount==patchVertexCount);
						else
							patchVertexCount = params.tessPatchVertCount;
						break;
					default:
						assert(false);
						break;
				}
				
				const bool iota = cpumb->getIndexType()==EIT_UNKNOWN||!cpumb->getIndexBufferBinding().buffer;
				if (iota)
					iotaLength = core::max(cpumb->getIndexCount(),iotaLength);
			}
			core::smart_refctd_ptr<ICPUBuffer> iotaUint32Buffer;
			if (iotaLength)
			{
				iotaUint32Buffer = ICPUBuffer::create({ sizeof(uint32_t)*iotaLength });
				auto ptr = reinterpret_cast<uint32_t*>(iotaUint32Buffer->getPointer());
				std::iota(ptr,ptr+iotaLength,0u);
			}
			// modify
			for (auto it=_begin; it!=_end; it++)
			{
				auto& cpumb = *it;

				const auto indexType = cpumb->getIndexType();
				auto indexCount = cpumb->getIndexCount();

				auto& params = cpumb->getPipeline()->getCachedCreationParams().primitiveAssembly;
				core::smart_refctd_ptr<ICPUBuffer> newIndexBuffer;

				void* correctlyOffsetIndexBufferPtr;

				const bool iota = indexType==EIT_UNKNOWN||!cpumb->getIndexBufferBinding().buffer;
				if (iota)
					correctlyOffsetIndexBufferPtr = iotaUint32Buffer->getPointer();
				else
					correctlyOffsetIndexBufferPtr = cpumb->getIndices();
				switch (params.primitiveType)
				{
					case EPT_LINE_STRIP:
						assert(_newPrimitiveType==EPT_LINE_LIST);
						newIndexBuffer = idxBufferFromLineStripsToLines(correctlyOffsetIndexBufferPtr,indexCount,iota ? EIT_32BIT:indexType,outIndexType);
						break;
					case EPT_TRIANGLE_STRIP:
						newIndexBuffer = idxBufferFromTriangleStripsToTriangles(correctlyOffsetIndexBufferPtr,indexCount,iota ? EIT_32BIT:indexType,outIndexType);
						break;
					case EPT_TRIANGLE_FAN:
						newIndexBuffer = idxBufferFromTrianglesFanToTriangles(correctlyOffsetIndexBufferPtr,indexCount,iota ? EIT_32BIT:indexType,outIndexType);
						break;
					default: // prim types match
						if (iota)
							newIndexBuffer = core::smart_refctd_ptr(iotaUint32Buffer);
						else if (indexType!=outIndexType)
						{
							if (indexType==EIT_16BIT)
							{
								auto inPtr = reinterpret_cast<const uint16_t*>(correctlyOffsetIndexBufferPtr);
								newIndexBuffer = ICPUBuffer::create({ sizeof(uint32_t)*indexCount });
								std::copy(inPtr,inPtr+indexCount,reinterpret_cast<uint32_t*>(newIndexBuffer->getPointer()));
							}
							else
							{
								auto inPtr = reinterpret_cast<const uint32_t*>(correctlyOffsetIndexBufferPtr);
								newIndexBuffer = ICPUBuffer::create({ sizeof(uint16_t)*indexCount });
								std::copy(inPtr,inPtr+indexCount,reinterpret_cast<uint16_t*>(newIndexBuffer->getPointer()));
							}
						}
						break;
				}
				if (newIndexBuffer)
				{
					cpumb->setIndexBufferBinding({0ull,std::move(newIndexBuffer)});
					cpumb->setIndexCount(indexCount);
				}
				cpumb->setIndexType(outIndexType);
				params.primitiveType = _newPrimitiveType;
			}
		}
#endif
};

#if 0

//! An interface for easy manipulation of meshes.
/** Scale, set alpha value, flip surfaces, and so on. This exists for fixing
problems with wrong imported or exported meshes quickly after loading. It is
not intended for doing mesh modifications and/or animations during runtime.
*/
class CMeshManipulator : public IMeshManipulator
{
		struct SAttrib
		{
			E_FORMAT type;
			E_FORMAT prevType;
			size_t size;
			uint32_t vaid;
			size_t offset;

			SAttrib() : type(EF_UNKNOWN), size(0), vaid(ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT) {}

			friend bool operator>(const SAttrib& _a, const SAttrib& _b) { return _a.size > _b.size; }
		};
		struct SAttribTypeChoice
		{
			E_FORMAT type;
		};

	public:
		static core::smart_refctd_ptr<ICPUMeshBuffer> createMeshBufferFetchOptimized(const ICPUMeshBuffer* _inbuffer);

		CQuantNormalCache* getQuantNormalCache() override { return &quantNormalCache; }
		CQuantQuaternionCache* getQuantQuaternionCache() override { return &quantQuaternionCache; }

	private:
		friend class IMeshManipulator;

		template<typename IdxT>
		static void _filterInvalidTriangles(ICPUMeshBuffer* _input);

		//! Meant to create 32bit index buffer from subrange of index buffer containing 16bit indices. Remember to set to index buffer offset to 0 after mapping buffer resulting from this function.
		static inline core::smart_refctd_ptr<ICPUBuffer> create32BitFrom16BitIdxBufferSubrange(const uint16_t* _in, uint32_t _idxCount)
		{
			if (!_in)
				return nullptr;

			auto out = ICPUBuffer::create({ sizeof(uint32_t) * _idxCount });

			auto* outPtr = reinterpret_cast<uint32_t*>(out->getPointer());

			for (uint32_t i=0u; i<_idxCount; ++i)
				outPtr[i] = _in[i];

			return out;
		}

		static core::vector<core::vectorSIMDf> findBetterFormatF(E_FORMAT* _outType, size_t* _outSize, E_FORMAT* _outPrevType, const ICPUMeshBuffer* _meshbuffer, uint32_t _attrId, const SErrorMetric& _errMetric, CQuantNormalCache& _cache);

		struct SIntegerAttr
		{
			uint32_t pointer[4];
		};
		static core::vector<SIntegerAttr> findBetterFormatI(E_FORMAT* _outType, size_t* _outSize, E_FORMAT* _outPrevType, const ICPUMeshBuffer* _meshbuffer, uint32_t _attrId, const SErrorMetric& _errMetric);

		//E_COMPONENT_TYPE getBestTypeF(bool _normalized, E_COMPONENTS_PER_ATTRIBUTE _cpa, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const float* _min, const float* _max) const;
		static E_FORMAT getBestTypeI(E_FORMAT _originalType, size_t* _outSize, const uint32_t* _min, const uint32_t* _max);
		static core::vector<SAttribTypeChoice> findTypesOfProperRangeF(E_FORMAT _type, size_t _sizeThreshold, const float* _min, const float* _max, const SErrorMetric& _errMetric);

		//! Calculates quantization errors and compares them with given epsilon.
		/** @returns false when first of calculated errors goes above epsilon or true if reached end without such. */
		static bool calcMaxQuantizationError(const SAttribTypeChoice& _srcType, const SAttribTypeChoice& _dstType, const core::vector<core::vectorSIMDf>& _data, const SErrorMetric& _errMetric, CQuantNormalCache& _cache);

		template<typename InType, typename OutType>
		static inline core::smart_refctd_ptr<ICPUBuffer> lineStripsToLines(const void* _input, uint32_t& _idxCount)
		{
			const auto outputSize = _idxCount = (_idxCount - 1) * 2;
			
			auto output = ICPUBuffer::create({ sizeof(OutType)*outputSize });
			const auto* iptr = reinterpret_cast<const InType*>(_input);
			auto* optr = reinterpret_cast<OutType*>(output->getPointer());
			for (uint32_t i = 0, j = 0; i < outputSize;)
			{
				optr[i++] = iptr[j++];
				optr[i++] = iptr[j];
			}
			return output;
		}

		template<typename InType, typename OutType>
		static inline core::smart_refctd_ptr<ICPUBuffer> triangleStripsToTriangles(const void* _input, uint32_t& _idxCount)
		{
			const auto outputSize = _idxCount = (_idxCount - 2) * 3;
			
			auto output = ICPUBuffer::create({ sizeof(OutType)*outputSize });
			const auto* iptr = reinterpret_cast<const InType*>(_input);
			auto* optr = reinterpret_cast<OutType*>(output->getPointer());
			for (uint32_t i = 0, j = 0; i < outputSize; j += 2)
			{
				optr[i++] = iptr[j + 0];
				optr[i++] = iptr[j + 1];
				optr[i++] = iptr[j + 2];
				if (i == outputSize)
					break;
				optr[i++] = iptr[j + 2];
				optr[i++] = iptr[j + 1];
				optr[i++] = iptr[j + 3];
			}
			return output;
		}

		template<typename InType, typename OutType>
		static inline core::smart_refctd_ptr<ICPUBuffer> trianglesFanToTriangles(const void* _input, uint32_t& _idxCount)
		{
			const auto outputSize = _idxCount = (_idxCount - 2) * 3;

			auto output = ICPUBuffer::create({ sizeof(OutType)*outputSize });
			const auto* iptr = reinterpret_cast<const InType*>(_input);
			auto* optr = reinterpret_cast<OutType*>(output->getPointer());
			for (uint32_t i = 0, j = 1; i < outputSize;)
			{
				optr[i++] = iptr[0];
				optr[i++] = iptr[j++];
				optr[i++] = iptr[j];
			}
			return output;
		}

	private:	
		CQuantNormalCache quantNormalCache;
		CQuantQuaternionCache quantQuaternionCache;
};
#endif

} // end namespace nbl::asset
#endif
