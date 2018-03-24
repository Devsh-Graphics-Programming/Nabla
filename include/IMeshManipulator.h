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
#ifndef NEW_MESHES
	    virtual void isolateAndExtractMeshBuffer(ICPUMeshBuffer* inbuffer, const bool& interleaved=true) const = 0;
#endif // NEW_MESHES
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
		virtual ICPUMeshBuffer* createMeshBufferWelded(ICPUMeshBuffer* inbuffer, const bool& makeNewMesh=false, float tolerance=core::ROUNDING_ERROR_f32) const = 0;

		virtual ICPUMeshBuffer* createOptimizedMeshBuffer(ICPUMeshBuffer* inbuffer) const = 0;

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

#ifndef NEW_MESHES
		//! Vertex cache optimization according to the Forsyth paper
		/** More information can be found at
		http://home.comcast.net/~tom_forsyth/papers/fast_vert_cache_opt.html

		The function is thread-safe (read: you can optimize several
		meshes in different threads).

		\param mesh Source mesh for the operation.
		\return A new mesh optimized for the vertex cache. */
        virtual ICPUMeshBuffer* createForsythOptimizedMeshBuffer(const ICPUMeshBuffer *meshbuffer) const = 0;
#endif // NEW_MESHES

protected:
};

} // end namespace scene
} // end namespace irr


#endif
