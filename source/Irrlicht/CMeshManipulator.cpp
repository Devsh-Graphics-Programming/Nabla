// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMeshManipulator.h"
#include "SMesh.h"
#include "IMeshBuffer.h"
#include "SAnimatedMesh.h"
#include "os.h"
#include "irrMap.h"
#include <vector>

namespace irr
{
namespace scene
{

std::vector<QuantizationCacheEntry2_10_10_10> normalCacheFor2_10_10_10Quant;


static inline core::vector3df getAngleWeight(const core::vector3df& v1,
		const core::vector3df& v2,
		const core::vector3df& v3)
{
	// Calculate this triangle's weight for each of its three vertices
	// start by calculating the lengths of its sides
	const f32 a = v2.getDistanceFromSQ(v3);
	const f32 asqrt = sqrtf(a);
	const f32 b = v1.getDistanceFromSQ(v3);
	const f32 bsqrt = sqrtf(b);
	const f32 c = v1.getDistanceFromSQ(v2);
	const f32 csqrt = sqrtf(c);

	// use them to find the angle at each vertex
	return core::vector3df(
		acosf((b + c - a) / (2.f * bsqrt * csqrt)),
		acosf((-b + c + a) / (2.f * asqrt * csqrt)),
		acosf((b - c + a) / (2.f * bsqrt * asqrt)));
}


//! Flips the direction of surfaces. Changes backfacing triangles to frontfacing
//! triangles and vice versa.
//! \param mesh: Mesh on which the operation is performed.
void CMeshManipulator::flipSurfaces(scene::ICPUMeshBuffer* inbuffer) const
{
	if (!inbuffer)
		return;

    const u32 idxcnt = inbuffer->getIndexCount();
    if (!inbuffer->getIndices())
        return;


    if (inbuffer->getIndexType() == video::EIT_16BIT)
    {
        uint16_t* idx = reinterpret_cast<uint16_t*>(inbuffer->getIndices());
        switch (inbuffer->getPrimitiveType())
        {
        case EPT_TRIANGLE_FAN:
            for (u32 i=1; i<idxcnt; i+=2)
            {
                const uint16_t tmp = idx[i];
                idx[i] = idx[i+1];
                idx[i+1] = tmp;
            }
            break;
        case EPT_TRIANGLE_STRIP:
            if (idxcnt%2) //odd
            {
                for (u32 i=0; i<(idxcnt>>1); i++)
                {
                    const uint16_t tmp = idx[i];
                    idx[i] = idx[idxcnt-1-i];
                    idx[idxcnt-1-i] = tmp;
                }
            }
            else //even
            {
                core::ICPUBuffer* newIndexBuffer = new core::ICPUBuffer(idxcnt*2+2);
                ((uint16_t*)newIndexBuffer->getPointer())[0] = idx[0];
                memcpy(((uint16_t*)newIndexBuffer->getPointer())+1,idx,idxcnt*2);
                inbuffer->setIndexCount(idxcnt+1);
                inbuffer->setIndexBufferOffset(0);
                inbuffer->getMeshDataAndFormat()->mapIndexBuffer(newIndexBuffer);
                newIndexBuffer->drop();
            }
            break;
        case EPT_TRIANGLES:
            for (u32 i=0; i<idxcnt; i+=3)
            {
                const uint16_t tmp = idx[i+1];
                idx[i+1] = idx[i+2];
                idx[i+2] = tmp;
            }
            break;
        }
    }
    else if (inbuffer->getIndexType() == video::EIT_32BIT)
    {
        uint32_t* idx = reinterpret_cast<uint32_t*>(inbuffer->getIndices());
        switch (inbuffer->getPrimitiveType())
        {
        case EPT_TRIANGLE_FAN:
            for (u32 i=1; i<idxcnt; i+=2)
            {
                const uint32_t tmp = idx[i];
                idx[i] = idx[i+1];
                idx[i+1] = tmp;
            }
            break;
        case EPT_TRIANGLE_STRIP:
            if (idxcnt%2) //odd
            {
                for (u32 i=0; i<(idxcnt>>1); i++)
                {
                    const uint32_t tmp = idx[i];
                    idx[i] = idx[idxcnt-1-i];
                    idx[idxcnt-1-i] = tmp;
                }
            }
            else //even
            {
                core::ICPUBuffer* newIndexBuffer = new core::ICPUBuffer(idxcnt*4+4);
                ((uint32_t*)newIndexBuffer->getPointer())[0] = idx[0];
                memcpy(((uint32_t*)newIndexBuffer->getPointer())+1,idx,idxcnt*4);
                inbuffer->setIndexCount(idxcnt+1);
                inbuffer->setIndexBufferOffset(0);
                inbuffer->getMeshDataAndFormat()->mapIndexBuffer(newIndexBuffer);
                newIndexBuffer->drop();
            }
            break;
        case EPT_TRIANGLES:
            for (u32 i=0; i<idxcnt; i+=3)
            {
                const uint32_t tmp = idx[i+1];
                idx[i+1] = idx[i+2];
                idx[i+2] = tmp;
            }
            break;
        }
    }
}


#ifndef NEW_MESHES
namespace
{
template <typename T>
void recalculateNormalsT(IMeshBuffer* buffer, bool smooth, bool angleWeighted)
{
	const u32 vtxcnt = buffer->getVertexCount();
	const u32 idxcnt = buffer->getIndexCount();
	const T* idx = reinterpret_cast<T*>(buffer->getIndices());

	if (!smooth)
	{
		for (u32 i=0; i<idxcnt; i+=3)
		{
			const core::vector3df& v1 = buffer->getPosition(idx[i+0]);
			const core::vector3df& v2 = buffer->getPosition(idx[i+1]);
			const core::vector3df& v3 = buffer->getPosition(idx[i+2]);
			const core::vector3df normal = core::plane3d<f32>(v1, v2, v3).Normal;
			buffer->getNormal(idx[i+0]) = normal;
			buffer->getNormal(idx[i+1]) = normal;
			buffer->getNormal(idx[i+2]) = normal;
		}
	}
	else
	{
		u32 i;

		for ( i = 0; i!= vtxcnt; ++i )
			buffer->getNormal(i).set(0.f, 0.f, 0.f);

		for ( i=0; i<idxcnt; i+=3)
		{
			const core::vector3df& v1 = buffer->getPosition(idx[i+0]);
			const core::vector3df& v2 = buffer->getPosition(idx[i+1]);
			const core::vector3df& v3 = buffer->getPosition(idx[i+2]);
			const core::vector3df normal = core::plane3d<f32>(v1, v2, v3).Normal;

			core::vector3df weight(1.f,1.f,1.f);
			if (angleWeighted)
				weight = irr::scene::getAngleWeight(v1,v2,v3); // writing irr::scene:: necessary for borland

			buffer->getNormal(idx[i+0]) += weight.X*normal;
			buffer->getNormal(idx[i+1]) += weight.Y*normal;
			buffer->getNormal(idx[i+2]) += weight.Z*normal;
		}

		for ( i = 0; i!= vtxcnt; ++i )
			buffer->getNormal(i).normalize();
	}
}
}


//! Recalculates all normals of the mesh buffer.
/** \param buffer: Mesh buffer on which the operation is performed. */
void CMeshManipulator::recalculateNormals(IMeshBuffer* buffer, bool smooth, bool angleWeighted) const
{
	if (!buffer)
		return;

	if (buffer->getIndexType()==video::EIT_16BIT)
		recalculateNormalsT<u16>(buffer, smooth, angleWeighted);
	else
		recalculateNormalsT<u32>(buffer, smooth, angleWeighted);
}


//! Recalculates all normals of the mesh.
//! \param mesh: Mesh on which the operation is performed.
void CMeshManipulator::recalculateNormals(scene::IMesh* mesh, bool smooth, bool angleWeighted) const
{
	if (!mesh)
		return;

	const u32 bcount = mesh->getMeshBufferCount();
	for ( u32 b=0; b<bcount; ++b)
		recalculateNormals(mesh->getMeshBuffer(b), smooth, angleWeighted);
}


namespace
{
void calculateTangents(
	core::vector3df& normal,
	core::vector3df& tangent,
	core::vector3df& binormal,
	const core::vector3df& vt1, const core::vector3df& vt2, const core::vector3df& vt3, // vertices
	const core::vector2df& tc1, const core::vector2df& tc2, const core::vector2df& tc3) // texture coords
{
	// choose one of them:
	//#define USE_NVIDIA_GLH_VERSION // use version used by nvidia in glh headers
	#define USE_IRR_VERSION

#ifdef USE_IRR_VERSION

	core::vector3df v1 = vt1 - vt2;
	core::vector3df v2 = vt3 - vt1;
	normal = v2.crossProduct(v1);
	normal.normalize();

	// binormal

	f32 deltaX1 = tc1.X - tc2.X;
	f32 deltaX2 = tc3.X - tc1.X;
	binormal = (v1 * deltaX2) - (v2 * deltaX1);
	binormal.normalize();

	// tangent

	f32 deltaY1 = tc1.Y - tc2.Y;
	f32 deltaY2 = tc3.Y - tc1.Y;
	tangent = (v1 * deltaY2) - (v2 * deltaY1);
	tangent.normalize();

	// adjust

	core::vector3df txb = tangent.crossProduct(binormal);
	if (txb.dotProduct(normal) < 0.0f)
	{
		tangent *= -1.0f;
		binormal *= -1.0f;
	}

#endif // USE_IRR_VERSION

#ifdef USE_NVIDIA_GLH_VERSION

	tangent.set(0,0,0);
	binormal.set(0,0,0);

	core::vector3df v1(vt2.X - vt1.X, tc2.X - tc1.X, tc2.Y - tc1.Y);
	core::vector3df v2(vt3.X - vt1.X, tc3.X - tc1.X, tc3.Y - tc1.Y);

	core::vector3df txb = v1.crossProduct(v2);
	if ( !core::iszero ( txb.X ) )
	{
		tangent.X  = -txb.Y / txb.X;
		binormal.X = -txb.Z / txb.X;
	}

	v1.X = vt2.Y - vt1.Y;
	v2.X = vt3.Y - vt1.Y;
	txb = v1.crossProduct(v2);

	if ( !core::iszero ( txb.X ) )
	{
		tangent.Y  = -txb.Y / txb.X;
		binormal.Y = -txb.Z / txb.X;
	}

	v1.X = vt2.Z - vt1.Z;
	v2.X = vt3.Z - vt1.Z;
	txb = v1.crossProduct(v2);

	if ( !core::iszero ( txb.X ) )
	{
		tangent.Z  = -txb.Y / txb.X;
		binormal.Z = -txb.Z / txb.X;
	}

	tangent.normalize();
	binormal.normalize();

	normal = tangent.crossProduct(binormal);
	normal.normalize();

	binormal = tangent.crossProduct(normal);
	binormal.normalize();

	core::plane3d<f32> pl(vt1, vt2, vt3);

	if(normal.dotProduct(pl.Normal) < 0.0f )
		normal *= -1.0f;

#endif // USE_NVIDIA_GLH_VERSION
}


//! Recalculates tangents for a tangent mesh buffer
template <typename T>
void recalculateTangentsT(IMeshBuffer* buffer, bool recalculateNormals, bool smooth, bool angleWeighted)
{
	if (!buffer || (buffer->getVertexType()!= video::EVT_TANGENTS))
		return;

	const u32 vtxCnt = buffer->getVertexCount();
	const u32 idxCnt = buffer->getIndexCount();

	T* idx = reinterpret_cast<T*>(buffer->getIndices());
	video::S3DVertexTangents* v =
		(video::S3DVertexTangents*)buffer->getVertices();

	if (smooth)
	{
		u32 i;

		for ( i = 0; i!= vtxCnt; ++i )
		{
			if (recalculateNormals)
				v[i].Normal.set( 0.f, 0.f, 0.f );
			v[i].Tangent.set( 0.f, 0.f, 0.f );
			v[i].Binormal.set( 0.f, 0.f, 0.f );
		}

		//Each vertex gets the sum of the tangents and binormals from the faces around it
		for ( i=0; i<idxCnt; i+=3)
		{
			// if this triangle is degenerate, skip it!
			if (v[idx[i+0]].Pos == v[idx[i+1]].Pos ||
				v[idx[i+0]].Pos == v[idx[i+2]].Pos ||
				v[idx[i+1]].Pos == v[idx[i+2]].Pos
				/*||
				v[idx[i+0]].TCoords == v[idx[i+1]].TCoords ||
				v[idx[i+0]].TCoords == v[idx[i+2]].TCoords ||
				v[idx[i+1]].TCoords == v[idx[i+2]].TCoords */
				)
				continue;

			//Angle-weighted normals look better, but are slightly more CPU intensive to calculate
			core::vector3df weight(1.f,1.f,1.f);
			if (angleWeighted)
				weight = irr::scene::getAngleWeight(v[i+0].Pos,v[i+1].Pos,v[i+2].Pos);	// writing irr::scene:: necessary for borland
			core::vector3df localNormal;
			core::vector3df localTangent;
			core::vector3df localBinormal;

			calculateTangents(
				localNormal,
				localTangent,
				localBinormal,
				v[idx[i+0]].Pos,
				v[idx[i+1]].Pos,
				v[idx[i+2]].Pos,
				v[idx[i+0]].TCoords,
				v[idx[i+1]].TCoords,
				v[idx[i+2]].TCoords);

			if (recalculateNormals)
				v[idx[i+0]].Normal += localNormal * weight.X;
			v[idx[i+0]].Tangent += localTangent * weight.X;
			v[idx[i+0]].Binormal += localBinormal * weight.X;

			calculateTangents(
				localNormal,
				localTangent,
				localBinormal,
				v[idx[i+1]].Pos,
				v[idx[i+2]].Pos,
				v[idx[i+0]].Pos,
				v[idx[i+1]].TCoords,
				v[idx[i+2]].TCoords,
				v[idx[i+0]].TCoords);

			if (recalculateNormals)
				v[idx[i+1]].Normal += localNormal * weight.Y;
			v[idx[i+1]].Tangent += localTangent * weight.Y;
			v[idx[i+1]].Binormal += localBinormal * weight.Y;

			calculateTangents(
				localNormal,
				localTangent,
				localBinormal,
				v[idx[i+2]].Pos,
				v[idx[i+0]].Pos,
				v[idx[i+1]].Pos,
				v[idx[i+2]].TCoords,
				v[idx[i+0]].TCoords,
				v[idx[i+1]].TCoords);

			if (recalculateNormals)
				v[idx[i+2]].Normal += localNormal * weight.Z;
			v[idx[i+2]].Tangent += localTangent * weight.Z;
			v[idx[i+2]].Binormal += localBinormal * weight.Z;
		}

		// Normalize the tangents and binormals
		if (recalculateNormals)
		{
			for ( i = 0; i!= vtxCnt; ++i )
				v[i].Normal.normalize();
		}
		for ( i = 0; i!= vtxCnt; ++i )
		{
			v[i].Tangent.normalize();
			v[i].Binormal.normalize();
		}
	}
	else
	{
		core::vector3df localNormal;
		for (u32 i=0; i<idxCnt; i+=3)
		{
			calculateTangents(
				localNormal,
				v[idx[i+0]].Tangent,
				v[idx[i+0]].Binormal,
				v[idx[i+0]].Pos,
				v[idx[i+1]].Pos,
				v[idx[i+2]].Pos,
				v[idx[i+0]].TCoords,
				v[idx[i+1]].TCoords,
				v[idx[i+2]].TCoords);
			if (recalculateNormals)
				v[idx[i+0]].Normal=localNormal;

			calculateTangents(
				localNormal,
				v[idx[i+1]].Tangent,
				v[idx[i+1]].Binormal,
				v[idx[i+1]].Pos,
				v[idx[i+2]].Pos,
				v[idx[i+0]].Pos,
				v[idx[i+1]].TCoords,
				v[idx[i+2]].TCoords,
				v[idx[i+0]].TCoords);
			if (recalculateNormals)
				v[idx[i+1]].Normal=localNormal;

			calculateTangents(
				localNormal,
				v[idx[i+2]].Tangent,
				v[idx[i+2]].Binormal,
				v[idx[i+2]].Pos,
				v[idx[i+0]].Pos,
				v[idx[i+1]].Pos,
				v[idx[i+2]].TCoords,
				v[idx[i+0]].TCoords,
				v[idx[i+1]].TCoords);
			if (recalculateNormals)
				v[idx[i+2]].Normal=localNormal;
		}
	}
}
}


//! Recalculates tangents for a tangent mesh buffer
void CMeshManipulator::recalculateTangents(IMeshBuffer* buffer, bool recalculateNormals, bool smooth, bool angleWeighted) const
{
	if (buffer && (buffer->getVertexType() == video::EVT_TANGENTS))
	{
		if (buffer->getIndexType() == video::EIT_16BIT)
			recalculateTangentsT<u16>(buffer, recalculateNormals, smooth, angleWeighted);
		else
			recalculateTangentsT<u32>(buffer, recalculateNormals, smooth, angleWeighted);
	}
}


//! Recalculates tangents for all tangent mesh buffers
void CMeshManipulator::recalculateTangents(IMesh* mesh, bool recalculateNormals, bool smooth, bool angleWeighted) const
{
	if (!mesh)
		return;

	const u32 meshBufferCount = mesh->getMeshBufferCount();
	for (u32 b=0; b<meshBufferCount; ++b)
	{
		recalculateTangents(mesh->getMeshBuffer(b), recalculateNormals, smooth, angleWeighted);
	}
}


namespace
{
//! Creates a planar texture mapping on the meshbuffer
template<typename T>
void makePlanarTextureMappingT(scene::IMeshBuffer* buffer, f32 resolution)
{
	u32 idxcnt = buffer->getIndexCount();
	T* idx = reinterpret_cast<T*>(buffer->getIndices());

	for (u32 i=0; i<idxcnt; i+=3)
	{
		core::plane3df p(buffer->getPosition(idx[i+0]), buffer->getPosition(idx[i+1]), buffer->getPosition(idx[i+2]));
		p.Normal.X = fabsf(p.Normal.X);
		p.Normal.Y = fabsf(p.Normal.Y);
		p.Normal.Z = fabsf(p.Normal.Z);
		// calculate planar mapping worldspace coordinates

		if (p.Normal.X > p.Normal.Y && p.Normal.X > p.Normal.Z)
		{
			for (u32 o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = buffer->getPosition(idx[i+o]).Y * resolution;
				buffer->getTCoords(idx[i+o]).Y = buffer->getPosition(idx[i+o]).Z * resolution;
			}
		}
		else
		if (p.Normal.Y > p.Normal.X && p.Normal.Y > p.Normal.Z)
		{
			for (u32 o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = buffer->getPosition(idx[i+o]).X * resolution;
				buffer->getTCoords(idx[i+o]).Y = buffer->getPosition(idx[i+o]).Z * resolution;
			}
		}
		else
		{
			for (u32 o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = buffer->getPosition(idx[i+o]).X * resolution;
				buffer->getTCoords(idx[i+o]).Y = buffer->getPosition(idx[i+o]).Y * resolution;
			}
		}
	}
}
}


//! Creates a planar texture mapping on the meshbuffer
void CMeshManipulator::makePlanarTextureMapping(scene::IMeshBuffer* buffer, f32 resolution) const
{
	if (!buffer)
		return;

	if (buffer->getIndexType()==video::EIT_16BIT)
		makePlanarTextureMappingT<u16>(buffer, resolution);
	else
		makePlanarTextureMappingT<u32>(buffer, resolution);
}


//! Creates a planar texture mapping on the mesh
void CMeshManipulator::makePlanarTextureMapping(scene::IMesh* mesh, f32 resolution) const
{
	if (!mesh)
		return;

	const u32 bcount = mesh->getMeshBufferCount();
	for ( u32 b=0; b<bcount; ++b)
	{
		makePlanarTextureMapping(mesh->getMeshBuffer(b), resolution);
	}
}


namespace
{
//! Creates a planar texture mapping on the meshbuffer
template <typename T>
void makePlanarTextureMappingT(scene::IMeshBuffer* buffer, f32 resolutionS, f32 resolutionT, u8 axis, const core::vector3df& offset)
{
	u32 idxcnt = buffer->getIndexCount();
	T* idx = reinterpret_cast<T*>(buffer->getIndices());

	for (u32 i=0; i<idxcnt; i+=3)
	{
		// calculate planar mapping worldspace coordinates
		if (axis==0)
		{
			for (u32 o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = 0.5f+(buffer->getPosition(idx[i+o]).Z + offset.Z) * resolutionS;
				buffer->getTCoords(idx[i+o]).Y = 0.5f-(buffer->getPosition(idx[i+o]).Y + offset.Y) * resolutionT;
			}
		}
		else if (axis==1)
		{
			for (u32 o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = 0.5f+(buffer->getPosition(idx[i+o]).X + offset.X) * resolutionS;
				buffer->getTCoords(idx[i+o]).Y = 1.f-(buffer->getPosition(idx[i+o]).Z + offset.Z) * resolutionT;
			}
		}
		else if (axis==2)
		{
			for (u32 o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = 0.5f+(buffer->getPosition(idx[i+o]).X + offset.X) * resolutionS;
				buffer->getTCoords(idx[i+o]).Y = 0.5f-(buffer->getPosition(idx[i+o]).Y + offset.Y) * resolutionT;
			}
		}
	}
}
}


//! Creates a planar texture mapping on the meshbuffer
void CMeshManipulator::makePlanarTextureMapping(scene::IMeshBuffer* buffer, f32 resolutionS, f32 resolutionT, u8 axis, const core::vector3df& offset) const
{
	if (!buffer)
		return;

	if (buffer->getIndexType()==video::EIT_16BIT)
		makePlanarTextureMappingT<u16>(buffer, resolutionS, resolutionT, axis, offset);
	else
		makePlanarTextureMappingT<u32>(buffer, resolutionS, resolutionT, axis, offset);
}


//! Creates a planar texture mapping on the mesh
void CMeshManipulator::makePlanarTextureMapping(scene::IMesh* mesh, f32 resolutionS, f32 resolutionT, u8 axis, const core::vector3df& offset) const
{
	if (!mesh)
		return;

	const u32 bcount = mesh->getMeshBufferCount();
	for ( u32 b=0; b<bcount; ++b)
	{
		makePlanarTextureMapping(mesh->getMeshBuffer(b), resolutionS, resolutionT, axis, offset);
	}
}
#endif // NEW_MESHES


//! Creates a copy of the mesh, which will only consist of unique primitives
ICPUMeshBuffer* CMeshManipulator::createMeshBufferUniquePrimitives(ICPUMeshBuffer* inbuffer) const
{
	if (!inbuffer)
		return 0;
    IMeshDataFormatDesc<core::ICPUBuffer>* oldDesc = inbuffer->getMeshDataAndFormat();
    if (!oldDesc)
        return 0;

    if (!inbuffer->getIndices())
    {
        inbuffer->grab();
        return inbuffer;
    }
    const uint32_t idxCnt = inbuffer->getIndexCount();
    ICPUMeshBuffer* clone = new ICPUMeshBuffer();
    clone->setBoundingBox(inbuffer->getBoundingBox());
    clone->setIndexCount(idxCnt);
    const E_PRIMITIVE_TYPE ept = inbuffer->getPrimitiveType();
    clone->setPrimitiveType(ept);
    clone->getMaterial() = inbuffer->getMaterial();

    ICPUMeshDataFormatDesc* desc = new ICPUMeshDataFormatDesc();
    clone->setMeshDataAndFormat(desc);
    desc->drop();

    size_t stride = 0;
    int32_t offset[EVAI_COUNT];
    size_t newAttribSizes[EVAI_COUNT];
    uint8_t* sourceBuffers[EVAI_COUNT] = {NULL};
    size_t sourceBufferStrides[EVAI_COUNT];
    for (size_t i=0; i<EVAI_COUNT; i++)
    {
        const core::ICPUBuffer* vbuf = oldDesc->getMappedBuffer((scene::E_VERTEX_ATTRIBUTE_ID)i);
        if (vbuf)
        {
            offset[i] = stride;
            newAttribSizes[i] = vertexAttrSize[oldDesc->getAttribType((E_VERTEX_ATTRIBUTE_ID)i)][oldDesc->getAttribComponentCount((E_VERTEX_ATTRIBUTE_ID)i)];
            stride += newAttribSizes[i];
            if (stride>=0xdeadbeefu)
            {
                clone->drop();
                return 0;
            }
            sourceBuffers[i] = (uint8_t*)vbuf->getPointer();
            sourceBuffers[i] += oldDesc->getMappedBufferOffset((E_VERTEX_ATTRIBUTE_ID)i);
            sourceBufferStrides[i] = oldDesc->getMappedBufferStride((E_VERTEX_ATTRIBUTE_ID)i);
        }
        else
            offset[i] = -1;
    }

    core::ICPUBuffer* vertexBuffer = new core::ICPUBuffer(stride*idxCnt);
    for (size_t i=0; i<EVAI_COUNT; i++)
    {
        if (offset[i]>=0)
            desc->mapVertexAttrBuffer(vertexBuffer,(E_VERTEX_ATTRIBUTE_ID)i,oldDesc->getAttribComponentCount((E_VERTEX_ATTRIBUTE_ID)i),oldDesc->getAttribType((E_VERTEX_ATTRIBUTE_ID)i),stride,offset[i]);
    }
    vertexBuffer->drop();

    uint8_t* destPointer = (uint8_t*)vertexBuffer->getPointer();
    if (inbuffer->getIndexType()==video::EIT_16BIT)
    {
        uint16_t* idx = reinterpret_cast<uint16_t*>(inbuffer->getIndices());
        for (uint64_t i=0; i<idxCnt; i++,idx++)
        for (size_t j=0; j<EVAI_COUNT; j++)
        {
            if (offset[j]<0)
                continue;

            memcpy(destPointer,sourceBuffers[j]+(int64_t(*idx)+inbuffer->getBaseVertex())*sourceBufferStrides[j],newAttribSizes[j]);
            destPointer += newAttribSizes[j];
        }
    }
    else if (inbuffer->getIndexType()==video::EIT_32BIT)
    {
        uint32_t* idx = reinterpret_cast<uint32_t*>(inbuffer->getIndices());
        for (uint64_t i=0; i<idxCnt; i++,idx++)
        for (size_t j=0; j<EVAI_COUNT; j++)
        {
            if (offset[j]<0)
                continue;

            memcpy(destPointer,sourceBuffers[j]+(int64_t(*idx)+inbuffer->getBaseVertex())*sourceBufferStrides[j],newAttribSizes[j]);
            destPointer += newAttribSizes[j];
        }
    }

	return clone;
}

size_t cmpfunc_vertsz;
int cmpfunc (const void * a, const void * b)
{
   return memcmp((uint8_t*)a+4,(uint8_t*)b+4,cmpfunc_vertsz);
}

//! Creates a copy of a mesh, which will have identical vertices welded together
ICPUMeshBuffer* CMeshManipulator::createMeshBufferWelded(ICPUMeshBuffer *inbuffer, const bool& makeNewMesh, f32 tolerance) const
{
    if (!inbuffer)
        return 0;
    IMeshDataFormatDesc<core::ICPUBuffer>* oldDesc = inbuffer->getMeshDataAndFormat();
    if (!oldDesc)
        return 0;

    bool bufferPresent[EVAI_COUNT];
    ICPUMeshDataFormatDesc* desc = NULL;
    if (makeNewMesh)
    {
        desc = new ICPUMeshDataFormatDesc();
        if (!desc)
            return 0;
    }

    size_t vertexAttrSize[EVAI_COUNT];
    size_t vertexSize = 0;
    for (size_t i=0; i<EVAI_COUNT; i++)
    {
        const core::ICPUBuffer* buf = oldDesc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)i);
        if (buf)
        {
            bufferPresent[i] = true;
            scene::E_COMPONENTS_PER_ATTRIBUTE componentCount = oldDesc->getAttribComponentCount((E_VERTEX_ATTRIBUTE_ID)i);
            scene::E_COMPONENT_TYPE componentType = oldDesc->getAttribType((E_VERTEX_ATTRIBUTE_ID)i);
            vertexAttrSize[i] = scene::vertexAttrSize[componentType][componentCount];
            vertexSize += vertexAttrSize[i];
            if (makeNewMesh)
            {
                desc->mapVertexAttrBuffer(  const_cast<core::ICPUBuffer*>(buf),(E_VERTEX_ATTRIBUTE_ID)i,
                                            componentCount,componentType,
                                            oldDesc->getMappedBufferStride((E_VERTEX_ATTRIBUTE_ID)i),oldDesc->getMappedBufferOffset((E_VERTEX_ATTRIBUTE_ID)i));
            }
        }
        else
            bufferPresent[i] = false;
    }
    cmpfunc_vertsz = vertexSize;

    size_t vertexCount = 0;
    video::E_INDEX_TYPE oldIndexType = video::EIT_UNKNOWN;
    if (oldDesc->getIndexBuffer())
    {
        if (inbuffer->getIndexType()==video::EIT_16BIT)
        {
            oldIndexType = video::EIT_16BIT;
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
            {
                size_t index = reinterpret_cast<const uint16_t*>(inbuffer->getIndices())[i];
                if (index>vertexCount)
                    vertexCount = index;
            }
            if (inbuffer->getIndexCount())
                vertexCount++;
        }
        else if (inbuffer->getIndexType()==video::EIT_32BIT)
        {
            oldIndexType = video::EIT_32BIT;
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
            {
                size_t index = reinterpret_cast<const uint32_t*>(inbuffer->getIndices())[i];
                if (index>vertexCount)
                    vertexCount = index;
            }
            if (inbuffer->getIndexCount())
                vertexCount++;
        }
        else
            vertexCount = inbuffer->getIndexCount();
    }
    else
        vertexCount = inbuffer->getIndexCount();

    if (vertexCount==0)
    {
        if (makeNewMesh)
            desc->drop();
        return 0;
    }

    // reset redirect list
    uint32_t* redirects = new uint32_t[vertexCount];

    uint32_t maxRedirect = 0;

    uint8_t* epicData = (uint8_t*)malloc((vertexSize+4)*vertexCount);
    for (size_t i=0; i < vertexCount; i++)
    {
        uint8_t* currentVertexPtr = epicData+i*(vertexSize+4);
        reinterpret_cast<uint32_t*>(currentVertexPtr)[0] = i;
        currentVertexPtr+=4;
        for (size_t k=0; k<EVAI_COUNT; k++)
        {
            if (!bufferPresent[k])
                continue;

            size_t stride = oldDesc->getMappedBufferStride((scene::E_VERTEX_ATTRIBUTE_ID)k);
            void* sourcePtr = inbuffer->getAttribPointer((scene::E_VERTEX_ATTRIBUTE_ID)k)+i*stride;
            memcpy(currentVertexPtr,sourcePtr,vertexAttrSize[k]);
            currentVertexPtr += vertexAttrSize[k];
        }
    }
    uint8_t* origData = (uint8_t*)malloc((vertexSize+4)*vertexCount);
    memcpy(origData,epicData,(vertexSize+4)*vertexCount);
    qsort(epicData, vertexCount, vertexSize+4, cmpfunc);
    for (size_t i=0; i<vertexCount; i++)
    {
        uint32_t redir;

        void* item = bsearch (origData+(vertexSize+4)*i, epicData, vertexCount, vertexSize+4, cmpfunc);
        if( item != NULL )
        {
            redir = *reinterpret_cast<uint32_t*>(item);
        }

        redirects[i] = redir;
        if (redir>maxRedirect)
            maxRedirect = redir;
    }
    free(origData);
    free(epicData);

    void* oldIndices = inbuffer->getIndices();
    ICPUMeshBuffer* clone = NULL;
    if (makeNewMesh)
    {
        clone = new ICPUMeshBuffer();
        if (!clone)
        {
            desc->drop();
            return 0;
        }
        clone->setBaseVertex(inbuffer->getBaseVertex());
        clone->setBoundingBox(inbuffer->getBoundingBox());
        clone->setIndexCount(inbuffer->getIndexCount());
        clone->setIndexType(maxRedirect>=0x10000u ? video::EIT_32BIT:video::EIT_16BIT);
        clone->setMeshDataAndFormat(desc);
        desc->drop();
        clone->setPrimitiveType(inbuffer->getPrimitiveType());
        clone->getMaterial() = inbuffer->getMaterial();

        core::ICPUBuffer* indexCpy = new core::ICPUBuffer((maxRedirect>=0x10000u ? 4:2)*inbuffer->getIndexCount());
        desc->mapIndexBuffer(indexCpy);
        indexCpy->drop();
    }
    else
    {
        if (!oldDesc->getIndexBuffer())
        {
            core::ICPUBuffer* indexCpy = new core::ICPUBuffer((maxRedirect>=0x10000u ? 4:2)*inbuffer->getIndexCount());
            oldDesc->mapIndexBuffer(indexCpy);
            indexCpy->drop();
        }
        inbuffer->setIndexType(maxRedirect>=0x10000u ? video::EIT_32BIT:video::EIT_16BIT);
    }


    if (oldIndexType==video::EIT_16BIT)
    {
        uint16_t* indicesIn = reinterpret_cast<uint16_t*>(oldIndices);
        if ((makeNewMesh ? clone:inbuffer)->getIndexType()==video::EIT_32BIT)
        {
            uint32_t* indicesOut = reinterpret_cast<uint32_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
        else if ((makeNewMesh ? clone:inbuffer)->getIndexType()==video::EIT_16BIT)
        {
            uint16_t* indicesOut = reinterpret_cast<uint16_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
    }
    else if (oldIndexType==video::EIT_32BIT)
    {
        uint32_t* indicesIn = reinterpret_cast<uint32_t*>(oldIndices);
        if ((makeNewMesh ? clone:inbuffer)->getIndexType()==video::EIT_32BIT)
        {
            uint32_t* indicesOut = reinterpret_cast<uint32_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
        else if ((makeNewMesh ? clone:inbuffer)->getIndexType()==video::EIT_16BIT)
        {
            uint16_t* indicesOut = reinterpret_cast<uint16_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
    }
    else if ((makeNewMesh ? clone:inbuffer)->getIndexType()==video::EIT_32BIT)
    {
        uint32_t* indicesOut = reinterpret_cast<uint32_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
        for (size_t i=0; i<inbuffer->getIndexCount(); i++)
            indicesOut[i] = redirects[i];
    }
    else if ((makeNewMesh ? clone:inbuffer)->getIndexType()==video::EIT_16BIT)
    {
        uint16_t* indicesOut = reinterpret_cast<uint16_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
        for (size_t i=0; i<inbuffer->getIndexCount(); i++)
            indicesOut[i] = redirects[i];
    }
    delete [] redirects;

    if (makeNewMesh)
        return clone;
    else
        return inbuffer;
}

template<>
bool IMeshManipulator::getPolyCount<core::ICPUBuffer>(uint32_t& outCount, IMeshBuffer<core::ICPUBuffer>* meshbuffer)
{
    outCount= 0;
    if (meshbuffer)
        return false;

    uint32_t trianglecount;

    switch (meshbuffer->getPrimitiveType())
    {
        case scene::EPT_POINTS:
            trianglecount = meshbuffer->getIndexCount();
            break;
        case scene::EPT_LINE_STRIP:
            trianglecount = meshbuffer->getIndexCount()-1;
            break;
        case scene::EPT_LINE_LOOP:
            trianglecount = meshbuffer->getIndexCount();
            break;
        case scene::EPT_LINES:
            trianglecount = meshbuffer->getIndexCount()/2;
            break;
        case scene::EPT_TRIANGLE_STRIP:
            trianglecount = meshbuffer->getIndexCount()-2;
            break;
        case scene::EPT_TRIANGLE_FAN:
            trianglecount = meshbuffer->getIndexCount()-2;
            break;
        case scene::EPT_TRIANGLES:
            trianglecount = meshbuffer->getIndexCount()/3;
            break;
    }

    outCount = trianglecount;
    return true;
}
template<>
bool IMeshManipulator::getPolyCount<video::IGPUBuffer>(uint32_t& outCount, IMeshBuffer<video::IGPUBuffer>* meshbuffer)
{
    outCount = 0;
    if (meshbuffer)
        return false;

    if (static_cast<IGPUMeshBuffer*>(meshbuffer)->isIndexCountGivenByXFormFeedback())
        return false;

    uint32_t trianglecount;

    switch (meshbuffer->getPrimitiveType())
    {
        case scene::EPT_POINTS:
            trianglecount = meshbuffer->getIndexCount();
            break;
        case scene::EPT_LINE_STRIP:
            trianglecount = meshbuffer->getIndexCount()-1;
            break;
        case scene::EPT_LINE_LOOP:
            trianglecount = meshbuffer->getIndexCount();
            break;
        case scene::EPT_LINES:
            trianglecount = meshbuffer->getIndexCount()/2;
            break;
        case scene::EPT_TRIANGLE_STRIP:
            trianglecount = meshbuffer->getIndexCount()-2;
            break;
        case scene::EPT_TRIANGLE_FAN:
            trianglecount = meshbuffer->getIndexCount()-2;
            break;
        case scene::EPT_TRIANGLES:
            trianglecount = meshbuffer->getIndexCount()/3;
            break;
    }

    outCount = trianglecount;
    return true;
}


//! Returns amount of polygons in mesh.
template<typename T>
bool IMeshManipulator::getPolyCount(uint32_t& outCount, scene::IMesh<T>* mesh)
{
    outCount = 0;
	if (!mesh)
		return false;

    bool retval = true;
	for (u32 g=0; g<mesh->getMeshBufferCount(); ++g)
    {
        uint32_t trianglecount;
        retval = retval&&getPolyCount(trianglecount,mesh->getMeshBuffer(g));
    }

	return retval;
}

template bool IMeshManipulator::getPolyCount<ICPUMeshBuffer>(uint32_t& outCount, IMesh<ICPUMeshBuffer>* mesh);
template bool IMeshManipulator::getPolyCount<IGPUMeshBuffer>(uint32_t& outCount, IMesh<IGPUMeshBuffer>* mesh);

#ifndef NEW_MESHES
//! Returns amount of polygons in mesh.
uint32_t IMeshManipulator::getPolyCount(scene::IAnimatedMesh* mesh)
{
	if (mesh && mesh->getFrameCount() != 0)
		return getPolyCount(mesh->getMesh(0));

	return 0;
}

namespace
{

struct vcache
{
	core::array<u32> tris;
	float score;
	s16 cachepos;
	u16 NumActiveTris;
};

struct tcache
{
	u16 ind[3];
	float score;
	bool drawn;
};

const u16 cachesize = 32;

float FindVertexScore(vcache *v)
{
	const float CacheDecayPower = 1.5f;
	const float LastTriScore = 0.75f;
	const float ValenceBoostScale = 2.0f;
	const float ValenceBoostPower = 0.5f;
	const float MaxSizeVertexCache = 32.0f;

	if (v->NumActiveTris == 0)
	{
		// No tri needs this vertex!
		return -1.0f;
	}

	float Score = 0.0f;
	int CachePosition = v->cachepos;
	if (CachePosition < 0)
	{
		// Vertex is not in FIFO cache - no score.
	}
	else
	{
		if (CachePosition < 3)
		{
			// This vertex was used in the last triangle,
			// so it has a fixed score.
			Score = LastTriScore;
		}
		else
		{
			// Points for being high in the cache.
			const float Scaler = 1.0f / (MaxSizeVertexCache - 3);
			Score = 1.0f - (CachePosition - 3) * Scaler;
			Score = powf(Score, CacheDecayPower);
		}
	}

	// Bonus points for having a low number of tris still to
	// use the vert, so we get rid of lone verts quickly.
	float ValenceBoost = powf(v->NumActiveTris,
				-ValenceBoostPower);
	Score += ValenceBoostScale * ValenceBoost;

	return Score;
}

/*
	A specialized LRU cache for the Forsyth algorithm.
*/

class f_lru
{

public:
	f_lru(vcache *v, tcache *t): vc(v), tc(t)
	{
		for (u16 i = 0; i < cachesize; i++)
		{
			cache[i] = -1;
		}
	}

	// Adds this vertex index and returns the highest-scoring triangle index
	u32 add(u16 vert, bool updatetris = false)
	{
		bool found = false;

		// Mark existing pos as empty
		for (u16 i = 0; i < cachesize; i++)
		{
			if (cache[i] == vert)
			{
				// Move everything down
				for (u16 j = i; j; j--)
				{
					cache[j] = cache[j - 1];
				}

				found = true;
				break;
			}
		}

		if (!found)
		{
			if (cache[cachesize-1] != -1)
				vc[cache[cachesize-1]].cachepos = -1;

			// Move everything down
			for (u16 i = cachesize - 1; i; i--)
			{
				cache[i] = cache[i - 1];
			}
		}

		cache[0] = vert;

		u32 highest = 0;
		float hiscore = 0;

		if (updatetris)
		{
			// Update cache positions
			for (u16 i = 0; i < cachesize; i++)
			{
				if (cache[i] == -1)
					break;

				vc[cache[i]].cachepos = i;
				vc[cache[i]].score = FindVertexScore(&vc[cache[i]]);
			}

			// Update triangle scores
			for (u16 i = 0; i < cachesize; i++)
			{
				if (cache[i] == -1)
					break;

				const u16 trisize = vc[cache[i]].tris.size();
				for (u16 t = 0; t < trisize; t++)
				{
					tcache *tri = &tc[vc[cache[i]].tris[t]];

					tri->score =
						vc[tri->ind[0]].score +
						vc[tri->ind[1]].score +
						vc[tri->ind[2]].score;

					if (tri->score > hiscore)
					{
						hiscore = tri->score;
						highest = vc[cache[i]].tris[t];
					}
				}
			}
		}

		return highest;
	}

private:
	s32 cache[cachesize];
	vcache *vc;
	tcache *tc;
};

} // end anonymous namespace

/**
Vertex cache optimization according to the Forsyth paper:
http://home.comcast.net/~tom_forsyth/papers/fast_vert_cache_opt.html

The function is thread-safe (read: you can optimize several meshes in different threads)

\param mesh Source mesh for the operation.  */
IMesh* CMeshManipulator::createForsythOptimizedMesh(const IMesh *mesh) const
{
	if (!mesh)
		return 0;

	SMesh *newmesh = new SMesh();
	newmesh->BoundingBox = mesh->getBoundingBox();

	const u32 mbcount = mesh->getMeshBufferCount();

	for (u32 b = 0; b < mbcount; ++b)
	{
		const IMeshBuffer *mb = mesh->getMeshBuffer(b);

		if (mb->getIndexType() != video::EIT_16BIT)
		{
			os::Printer::log("Cannot optimize a mesh with 32bit indices", ELL_ERROR);
			newmesh->drop();
			return 0;
		}

		const u32 icount = mb->getIndexCount();
		const u32 tcount = icount / 3;
		const u32 vcount = mb->getVertexCount();
		const u16* ind = reinterpret_cast<const u16*>(mb->getIndices());

		vcache *vc = new vcache[vcount];
		tcache *tc = new tcache[tcount];

		f_lru lru(vc, tc);

		// init
		for (u16 i = 0; i < vcount; i++)
		{
			vc[i].score = 0;
			vc[i].cachepos = -1;
			vc[i].NumActiveTris = 0;
		}

		// First pass: count how many times a vert is used
		for (u32 i = 0; i < icount; i += 3)
		{
			vc[ind[i]].NumActiveTris++;
			vc[ind[i + 1]].NumActiveTris++;
			vc[ind[i + 2]].NumActiveTris++;

			const u32 tri_ind = i/3;
			tc[tri_ind].ind[0] = ind[i];
			tc[tri_ind].ind[1] = ind[i + 1];
			tc[tri_ind].ind[2] = ind[i + 2];
		}

		// Second pass: list of each triangle
		for (u32 i = 0; i < tcount; i++)
		{
			vc[tc[i].ind[0]].tris.push_back(i);
			vc[tc[i].ind[1]].tris.push_back(i);
			vc[tc[i].ind[2]].tris.push_back(i);

			tc[i].drawn = false;
		}

		// Give initial scores
		for (u16 i = 0; i < vcount; i++)
		{
			vc[i].score = FindVertexScore(&vc[i]);
		}
		for (u32 i = 0; i < tcount; i++)
		{
			tc[i].score =
					vc[tc[i].ind[0]].score +
					vc[tc[i].ind[1]].score +
					vc[tc[i].ind[2]].score;
		}

		switch(mb->getVertexType())
		{
			case video::EVT_STANDARD:
			{
				video::S3DVertex *v = (video::S3DVertex *) mb->getVertices();

				SMeshBuffer *buf = new SMeshBuffer();
				buf->Material = mb->getMaterial();

				buf->Vertices.reallocate(vcount);
				buf->Indices.reallocate(icount);

				core::map<const video::S3DVertex, const u16> sind; // search index for fast operation
				typedef core::map<const video::S3DVertex, const u16>::Node snode;

				// Main algorithm
				u32 highest = 0;
				u32 drawcalls = 0;
				for (;;)
				{
					if (tc[highest].drawn)
					{
						bool found = false;
						float hiscore = 0;
						for (u32 t = 0; t < tcount; t++)
						{
							if (!tc[t].drawn)
							{
								if (tc[t].score > hiscore)
								{
									highest = t;
									hiscore = tc[t].score;
									found = true;
								}
							}
						}
						if (!found)
							break;
					}

					// Output the best triangle
					u16 newind = buf->Vertices.size();

					snode *s = sind.find(v[tc[highest].ind[0]]);

					if (!s)
					{
						buf->Vertices.push_back(v[tc[highest].ind[0]]);
						buf->Indices.push_back(newind);
						sind.insert(v[tc[highest].ind[0]], newind);
						newind++;
					}
					else
					{
						buf->Indices.push_back(s->getValue());
					}

					s = sind.find(v[tc[highest].ind[1]]);

					if (!s)
					{
						buf->Vertices.push_back(v[tc[highest].ind[1]]);
						buf->Indices.push_back(newind);
						sind.insert(v[tc[highest].ind[1]], newind);
						newind++;
					}
					else
					{
						buf->Indices.push_back(s->getValue());
					}

					s = sind.find(v[tc[highest].ind[2]]);

					if (!s)
					{
						buf->Vertices.push_back(v[tc[highest].ind[2]]);
						buf->Indices.push_back(newind);
						sind.insert(v[tc[highest].ind[2]], newind);
					}
					else
					{
						buf->Indices.push_back(s->getValue());
					}

					vc[tc[highest].ind[0]].NumActiveTris--;
					vc[tc[highest].ind[1]].NumActiveTris--;
					vc[tc[highest].ind[2]].NumActiveTris--;

					tc[highest].drawn = true;

					for (u16 j = 0; j < 3; j++)
					{
						vcache *vert = &vc[tc[highest].ind[j]];
						for (u16 t = 0; t < vert->tris.size(); t++)
						{
							if (highest == vert->tris[t])
							{
								vert->tris.erase(t);
								break;
							}
						}
					}

					lru.add(tc[highest].ind[0]);
					lru.add(tc[highest].ind[1]);
					highest = lru.add(tc[highest].ind[2], true);
					drawcalls++;
				}

				buf->setBoundingBox(mb->getBoundingBox());
				newmesh->addMeshBuffer(buf);
				buf->drop();
			}
			break;
			case video::EVT_2TCOORDS:
			{
				video::S3DVertex2TCoords *v = (video::S3DVertex2TCoords *) mb->getVertices();

				SMeshBufferLightMap *buf = new SMeshBufferLightMap();
				buf->Material = mb->getMaterial();

				buf->Vertices.reallocate(vcount);
				buf->Indices.reallocate(icount);

				core::map<const video::S3DVertex2TCoords, const u16> sind; // search index for fast operation
				typedef core::map<const video::S3DVertex2TCoords, const u16>::Node snode;

				// Main algorithm
				u32 highest = 0;
				u32 drawcalls = 0;
				for (;;)
				{
					if (tc[highest].drawn)
					{
						bool found = false;
						float hiscore = 0;
						for (u32 t = 0; t < tcount; t++)
						{
							if (!tc[t].drawn)
							{
								if (tc[t].score > hiscore)
								{
									highest = t;
									hiscore = tc[t].score;
									found = true;
								}
							}
						}
						if (!found)
							break;
					}

					// Output the best triangle
					u16 newind = buf->Vertices.size();

					snode *s = sind.find(v[tc[highest].ind[0]]);

					if (!s)
					{
						buf->Vertices.push_back(v[tc[highest].ind[0]]);
						buf->Indices.push_back(newind);
						sind.insert(v[tc[highest].ind[0]], newind);
						newind++;
					}
					else
					{
						buf->Indices.push_back(s->getValue());
					}

					s = sind.find(v[tc[highest].ind[1]]);

					if (!s)
					{
						buf->Vertices.push_back(v[tc[highest].ind[1]]);
						buf->Indices.push_back(newind);
						sind.insert(v[tc[highest].ind[1]], newind);
						newind++;
					}
					else
					{
						buf->Indices.push_back(s->getValue());
					}

					s = sind.find(v[tc[highest].ind[2]]);

					if (!s)
					{
						buf->Vertices.push_back(v[tc[highest].ind[2]]);
						buf->Indices.push_back(newind);
						sind.insert(v[tc[highest].ind[2]], newind);
					}
					else
					{
						buf->Indices.push_back(s->getValue());
					}

					vc[tc[highest].ind[0]].NumActiveTris--;
					vc[tc[highest].ind[1]].NumActiveTris--;
					vc[tc[highest].ind[2]].NumActiveTris--;

					tc[highest].drawn = true;

					for (u16 j = 0; j < 3; j++)
					{
						vcache *vert = &vc[tc[highest].ind[j]];
						for (u16 t = 0; t < vert->tris.size(); t++)
						{
							if (highest == vert->tris[t])
							{
								vert->tris.erase(t);
								break;
							}
						}
					}

					lru.add(tc[highest].ind[0]);
					lru.add(tc[highest].ind[1]);
					highest = lru.add(tc[highest].ind[2]);
					drawcalls++;
				}

				buf->setBoundingBox(mb->getBoundingBox());
				newmesh->addMeshBuffer(buf);
				buf->drop();

			}
			break;
			case video::EVT_TANGENTS:
			{
				video::S3DVertexTangents *v = (video::S3DVertexTangents *) mb->getVertices();

				SMeshBufferTangents *buf = new SMeshBufferTangents();
				buf->Material = mb->getMaterial();

				buf->Vertices.reallocate(vcount);
				buf->Indices.reallocate(icount);

				core::map<const video::S3DVertexTangents, const u16> sind; // search index for fast operation
				typedef core::map<const video::S3DVertexTangents, const u16>::Node snode;

				// Main algorithm
				u32 highest = 0;
				u32 drawcalls = 0;
				for (;;)
				{
					if (tc[highest].drawn)
					{
						bool found = false;
						float hiscore = 0;
						for (u32 t = 0; t < tcount; t++)
						{
							if (!tc[t].drawn)
							{
								if (tc[t].score > hiscore)
								{
									highest = t;
									hiscore = tc[t].score;
									found = true;
								}
							}
						}
						if (!found)
							break;
					}

					// Output the best triangle
					u16 newind = buf->Vertices.size();

					snode *s = sind.find(v[tc[highest].ind[0]]);

					if (!s)
					{
						buf->Vertices.push_back(v[tc[highest].ind[0]]);
						buf->Indices.push_back(newind);
						sind.insert(v[tc[highest].ind[0]], newind);
						newind++;
					}
					else
					{
						buf->Indices.push_back(s->getValue());
					}

					s = sind.find(v[tc[highest].ind[1]]);

					if (!s)
					{
						buf->Vertices.push_back(v[tc[highest].ind[1]]);
						buf->Indices.push_back(newind);
						sind.insert(v[tc[highest].ind[1]], newind);
						newind++;
					}
					else
					{
						buf->Indices.push_back(s->getValue());
					}

					s = sind.find(v[tc[highest].ind[2]]);

					if (!s)
					{
						buf->Vertices.push_back(v[tc[highest].ind[2]]);
						buf->Indices.push_back(newind);
						sind.insert(v[tc[highest].ind[2]], newind);
					}
					else
					{
						buf->Indices.push_back(s->getValue());
					}

					vc[tc[highest].ind[0]].NumActiveTris--;
					vc[tc[highest].ind[1]].NumActiveTris--;
					vc[tc[highest].ind[2]].NumActiveTris--;

					tc[highest].drawn = true;

					for (u16 j = 0; j < 3; j++)
					{
						vcache *vert = &vc[tc[highest].ind[j]];
						for (u16 t = 0; t < vert->tris.size(); t++)
						{
							if (highest == vert->tris[t])
							{
								vert->tris.erase(t);
								break;
							}
						}
					}

					lru.add(tc[highest].ind[0]);
					lru.add(tc[highest].ind[1]);
					highest = lru.add(tc[highest].ind[2]);
					drawcalls++;
				}

				buf->setBoundingBox(mb->getBoundingBox());
				newmesh->addMeshBuffer(buf);
				buf->drop();
			}
			break;
		}

		delete [] vc;
		delete [] tc;

	} // for each meshbuffer

	return newmesh;
}
#endif // NEW_MESHES

} // end namespace scene
} // end namespace irr

