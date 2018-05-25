// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_MESH_MANIPULATOR_H_INCLUDED__
#define __I_MESH_MANIPULATOR_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "IReferenceCounted.h"
#include "vector3d.h"
#include "aabbox3d.h"
#include "matrix4.h"
#include "IAnimatedMesh.h"
#include "IMeshBuffer.h"
#include "SVertexManipulator.h"
#include "SMesh.h"

namespace irr
{
namespace scene
{
	//! An interface for easy manipulation of meshes.
	/** Scale, set alpha value, flip surfaces, and so on. This exists for
	fixing problems with wrong imported or exported meshes quickly after
	loading. It is not intended for doing mesh modifications and/or
	animations during runtime.
	*/
	class IMeshManipulator : public virtual IReferenceCounted
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
	public:
		//! Flips the direction of surfaces.
		/** Changes backfacing triangles to frontfacing
		triangles and vice versa.
		\param mesh Mesh on which the operation is performed. */
		virtual void flipSurfaces(ICPUMeshBuffer* inbuffer) const = 0;

		//! Creates a copy of a mesh with all vertices unwelded
		/** \param mesh Input mesh
		\return Mesh consisting only of unique faces. All vertices
		which were previously shared are now duplicated. If you no
		longer need the cloned mesh, you should call IMesh::drop(). See
		IReferenceCounted::drop() for more information. */
		virtual ICPUMeshBuffer* createMeshBufferUniquePrimitives(ICPUMeshBuffer* inbuffer) const = 0;

		//! Creates a copy of a mesh with vertices welded
		/** \param mesh Input mesh
		\param tolerance The threshold for vertex comparisons.
		\return Mesh without redundant vertices. If you no longer need
		the cloned mesh, you should call IMesh::drop(). See
		IReferenceCounted::drop() for more information. */
		virtual ICPUMeshBuffer* createMeshBufferWelded(ICPUMeshBuffer* inbuffer, const bool& reduceIdxBufSize = false, const bool& makeNewMesh=false, float tolerance=core::ROUNDING_ERROR_f32) const = 0;

		//! Throws meshbuffer into full optimizing pipeline consisting of: vertices welding, z-buffer optimization, vertex cache optimization (Forsyth's algorithm), fetch optimization and attributes requantization. A new meshbuffer is created unless given meshbuffer doesn't own (getMeshDataAndFormat()==NULL) a data format descriptor.
		/**@return A new meshbuffer or NULL if an error occured. */
		virtual ICPUMeshBuffer* createOptimizedMeshBuffer(const ICPUMeshBuffer* inbuffer, const SErrorMetric* _requantErrMetric) const = 0;

		//! Requantizes vertex attributes to the smallest possible types taking into account values of the attribute under consideration. A brand new vertex buffer is created and attributes are going to be interleaved in single buffer.
		/**
			The function tests type's range and precision loss after eventual requantization. The latter is performed in one of several possible methods specified
		in array parameter. By this paramater user can define method of comparison (shall depend on what the attribute's data represents) and epsilon (i.e. precision error tolerance).
		@param _meshbuffer Input meshbuffer that is to be requantized.
		@param _errMetric Array of structs defining methods of error metrics. The array must be of EVAI_COUNT length since each index of the array directly corresponds to attribute's id.
		*/
		virtual void requantizeMeshBuffer(ICPUMeshBuffer* _meshbuffer, const SErrorMetric* _errMetric) const = 0;

		virtual ICPUMeshBuffer* createMeshBufferDuplicate(const ICPUMeshBuffer* _src) const = 0;

        //! Creates new index buffer with invalid triangles removed.
        /**
        Invalid triangle is such consisting of two or more same indices.
        @param _input Input index buffer.
        @param _idxType Type of indices in the index buffer.
        @returns New index buffer or nullptr if input indices were of unknown type or _input was nullptr.
        */
        virtual void filterInvalidTriangles(ICPUMeshBuffer* _input) const = 0;

		//! Get amount of polygons in mesh buffer.
		/** \param meshbuffer Input mesh buffer
		\param Outputted Number of polygons in mesh buffer, if successful.
		\return If successfully can provide information, i.e. if XFormFeedback is providing PolyCount we dont know how many there are */
		template<typename T>
		static bool getPolyCount(uint32_t& outCount, IMeshBuffer<T>* meshbuffer);

		//! Get amount of polygons in mesh.
		/** \param meshbuffer Input mesh
		\param Outputted Number of polygons in mesh, if successful.
		\return If successfully can provide information, i.e. if XFormFeedback is providing PolyCount we dont know how many there are */
		template<typename T>
		static bool getPolyCount(uint32_t& outCount, IMesh<T>* mesh);
protected:
};

} // end namespace scene
} // end namespace irr


#endif
