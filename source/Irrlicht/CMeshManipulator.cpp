// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMeshManipulator.h"

#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>

#include "SMesh.h"
#include "IMeshBuffer.h"
#include "SAnimatedMesh.h"
#include "os.h"
#include "CForsythVertexCacheOptimizer.h"
#include "SSkinMeshBuffer.h"

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
	const float a = v2.getDistanceFromSQ(v3);
	const float asqrt = sqrtf(a);
	const float b = v1.getDistanceFromSQ(v3);
	const float bsqrt = sqrtf(b);
	const float c = v1.getDistanceFromSQ(v2);
	const float csqrt = sqrtf(c);

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

    const uint32_t idxcnt = inbuffer->getIndexCount();
    if (!inbuffer->getIndices())
        return;


    if (inbuffer->getIndexType() == video::EIT_16BIT)
    {
        uint16_t* idx = reinterpret_cast<uint16_t*>(inbuffer->getIndices());
        switch (inbuffer->getPrimitiveType())
        {
        case EPT_TRIANGLE_FAN:
            for (uint32_t i=1; i<idxcnt; i+=2)
            {
                const uint16_t tmp = idx[i];
                idx[i] = idx[i+1];
                idx[i+1] = tmp;
            }
            break;
        case EPT_TRIANGLE_STRIP:
            if (idxcnt%2) //odd
            {
                for (uint32_t i=0; i<(idxcnt>>1); i++)
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
            for (uint32_t i=0; i<idxcnt; i+=3)
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
            for (uint32_t i=1; i<idxcnt; i+=2)
            {
                const uint32_t tmp = idx[i];
                idx[i] = idx[i+1];
                idx[i+1] = tmp;
            }
            break;
        case EPT_TRIANGLE_STRIP:
            if (idxcnt%2) //odd
            {
                for (uint32_t i=0; i<(idxcnt>>1); i++)
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
            for (uint32_t i=0; i<idxcnt; i+=3)
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
	const uint32_t vtxcnt = buffer->getVertexCount();
	const uint32_t idxcnt = buffer->getIndexCount();
	const T* idx = reinterpret_cast<T*>(buffer->getIndices());

	if (!smooth)
	{
		for (uint32_t i=0; i<idxcnt; i+=3)
		{
			const core::vector3df& v1 = buffer->getPosition(idx[i+0]);
			const core::vector3df& v2 = buffer->getPosition(idx[i+1]);
			const core::vector3df& v3 = buffer->getPosition(idx[i+2]);
			const core::vector3df normal = core::plane3d<float>(v1, v2, v3).Normal;
			buffer->getNormal(idx[i+0]) = normal;
			buffer->getNormal(idx[i+1]) = normal;
			buffer->getNormal(idx[i+2]) = normal;
		}
	}
	else
	{
		uint32_t i;

		for ( i = 0; i!= vtxcnt; ++i )
			buffer->getNormal(i).set(0.f, 0.f, 0.f);

		for ( i=0; i<idxcnt; i+=3)
		{
			const core::vector3df& v1 = buffer->getPosition(idx[i+0]);
			const core::vector3df& v2 = buffer->getPosition(idx[i+1]);
			const core::vector3df& v3 = buffer->getPosition(idx[i+2]);
			const core::vector3df normal = core::plane3d<float>(v1, v2, v3).Normal;

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
		recalculateNormalsT<uint16_t>(buffer, smooth, angleWeighted);
	else
		recalculateNormalsT<uint32_t>(buffer, smooth, angleWeighted);
}


//! Recalculates all normals of the mesh.
//! \param mesh: Mesh on which the operation is performed.
void CMeshManipulator::recalculateNormals(scene::IMesh* mesh, bool smooth, bool angleWeighted) const
{
	if (!mesh)
		return;

	const uint32_t bcount = mesh->getMeshBufferCount();
	for ( uint32_t b=0; b<bcount; ++b)
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

	float deltaX1 = tc1.X - tc2.X;
	float deltaX2 = tc3.X - tc1.X;
	binormal = (v1 * deltaX2) - (v2 * deltaX1);
	binormal.normalize();

	// tangent

	float deltaY1 = tc1.Y - tc2.Y;
	float deltaY2 = tc3.Y - tc1.Y;
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

	core::plane3d<float> pl(vt1, vt2, vt3);

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

	const uint32_t vtxCnt = buffer->getVertexCount();
	const uint32_t idxCnt = buffer->getIndexCount();

	T* idx = reinterpret_cast<T*>(buffer->getIndices());
	video::S3DVertexTangents* v =
		(video::S3DVertexTangents*)buffer->getVertices();

	if (smooth)
	{
		uint32_t i;

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
		for (uint32_t i=0; i<idxCnt; i+=3)
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
			recalculateTangentsT<uint16_t>(buffer, recalculateNormals, smooth, angleWeighted);
		else
			recalculateTangentsT<uint32_t>(buffer, recalculateNormals, smooth, angleWeighted);
	}
}


//! Recalculates tangents for all tangent mesh buffers
void CMeshManipulator::recalculateTangents(IMesh* mesh, bool recalculateNormals, bool smooth, bool angleWeighted) const
{
	if (!mesh)
		return;

	const uint32_t meshBufferCount = mesh->getMeshBufferCount();
	for (uint32_t b=0; b<meshBufferCount; ++b)
	{
		recalculateTangents(mesh->getMeshBuffer(b), recalculateNormals, smooth, angleWeighted);
	}
}


namespace
{
//! Creates a planar texture mapping on the meshbuffer
template<typename T>
void makePlanarTextureMappingT(scene::IMeshBuffer* buffer, float resolution)
{
	uint32_t idxcnt = buffer->getIndexCount();
	T* idx = reinterpret_cast<T*>(buffer->getIndices());

	for (uint32_t i=0; i<idxcnt; i+=3)
	{
		core::plane3df p(buffer->getPosition(idx[i+0]), buffer->getPosition(idx[i+1]), buffer->getPosition(idx[i+2]));
		p.Normal.X = fabsf(p.Normal.X);
		p.Normal.Y = fabsf(p.Normal.Y);
		p.Normal.Z = fabsf(p.Normal.Z);
		// calculate planar mapping worldspace coordinates

		if (p.Normal.X > p.Normal.Y && p.Normal.X > p.Normal.Z)
		{
			for (uint32_t o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = buffer->getPosition(idx[i+o]).Y * resolution;
				buffer->getTCoords(idx[i+o]).Y = buffer->getPosition(idx[i+o]).Z * resolution;
			}
		}
		else
		if (p.Normal.Y > p.Normal.X && p.Normal.Y > p.Normal.Z)
		{
			for (uint32_t o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = buffer->getPosition(idx[i+o]).X * resolution;
				buffer->getTCoords(idx[i+o]).Y = buffer->getPosition(idx[i+o]).Z * resolution;
			}
		}
		else
		{
			for (uint32_t o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = buffer->getPosition(idx[i+o]).X * resolution;
				buffer->getTCoords(idx[i+o]).Y = buffer->getPosition(idx[i+o]).Y * resolution;
			}
		}
	}
}
}


//! Creates a planar texture mapping on the meshbuffer
void CMeshManipulator::makePlanarTextureMapping(scene::IMeshBuffer* buffer, float resolution) const
{
	if (!buffer)
		return;

	if (buffer->getIndexType()==video::EIT_16BIT)
		makePlanarTextureMappingT<uint16_t>(buffer, resolution);
	else
		makePlanarTextureMappingT<uint32_t>(buffer, resolution);
}


//! Creates a planar texture mapping on the mesh
void CMeshManipulator::makePlanarTextureMapping(scene::IMesh* mesh, float resolution) const
{
	if (!mesh)
		return;

	const uint32_t bcount = mesh->getMeshBufferCount();
	for ( uint32_t b=0; b<bcount; ++b)
	{
		makePlanarTextureMapping(mesh->getMeshBuffer(b), resolution);
	}
}


namespace
{
//! Creates a planar texture mapping on the meshbuffer
template <typename T>
void makePlanarTextureMappingT(scene::IMeshBuffer* buffer, float resolutionS, float resolutionT, uint8_t axis, const core::vector3df& offset)
{
	uint32_t idxcnt = buffer->getIndexCount();
	T* idx = reinterpret_cast<T*>(buffer->getIndices());

	for (uint32_t i=0; i<idxcnt; i+=3)
	{
		// calculate planar mapping worldspace coordinates
		if (axis==0)
		{
			for (uint32_t o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = 0.5f+(buffer->getPosition(idx[i+o]).Z + offset.Z) * resolutionS;
				buffer->getTCoords(idx[i+o]).Y = 0.5f-(buffer->getPosition(idx[i+o]).Y + offset.Y) * resolutionT;
			}
		}
		else if (axis==1)
		{
			for (uint32_t o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = 0.5f+(buffer->getPosition(idx[i+o]).X + offset.X) * resolutionS;
				buffer->getTCoords(idx[i+o]).Y = 1.f-(buffer->getPosition(idx[i+o]).Z + offset.Z) * resolutionT;
			}
		}
		else if (axis==2)
		{
			for (uint32_t o=0; o!=3; ++o)
			{
				buffer->getTCoords(idx[i+o]).X = 0.5f+(buffer->getPosition(idx[i+o]).X + offset.X) * resolutionS;
				buffer->getTCoords(idx[i+o]).Y = 0.5f-(buffer->getPosition(idx[i+o]).Y + offset.Y) * resolutionT;
			}
		}
	}
}
}


//! Creates a planar texture mapping on the meshbuffer
void CMeshManipulator::makePlanarTextureMapping(scene::IMeshBuffer* buffer, float resolutionS, float resolutionT, uint8_t axis, const core::vector3df& offset) const
{
	if (!buffer)
		return;

	if (buffer->getIndexType()==video::EIT_16BIT)
		makePlanarTextureMappingT<uint16_t>(buffer, resolutionS, resolutionT, axis, offset);
	else
		makePlanarTextureMappingT<uint32_t>(buffer, resolutionS, resolutionT, axis, offset);
}


//! Creates a planar texture mapping on the mesh
void CMeshManipulator::makePlanarTextureMapping(scene::IMesh* mesh, float resolutionS, float resolutionT, uint8_t axis, const core::vector3df& offset) const
{
	if (!mesh)
		return;

	const uint32_t bcount = mesh->getMeshBufferCount();
	for ( uint32_t b=0; b<bcount; ++b)
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
ICPUMeshBuffer* CMeshManipulator::createMeshBufferWelded(ICPUMeshBuffer *inbuffer, const bool& makeNewMesh, float tolerance) const
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

ICPUMeshBuffer* CMeshManipulator::createOptimizedMeshBuffer(ICPUMeshBuffer* _inbuffer) const
{
	if (!_inbuffer->getMeshDataAndFormat())
		return NULL;

	ICPUMeshBuffer* outbuffer;
	if (SCPUSkinMeshBuffer* inb = dynamic_cast<SCPUSkinMeshBuffer*>(_inbuffer))
	{
		outbuffer = new SCPUSkinMeshBuffer();
		SCPUSkinMeshBuffer* outb = dynamic_cast<SCPUSkinMeshBuffer*>(outbuffer);
		outb->setIndexRange(inb->getIndexMinBound(), inb->getIndexMaxBound());
		outb->setMaxVertexBoneInfluences(inb->getMaxVertexBoneInfluences());
	}
	else
		outbuffer = new ICPUMeshBuffer();

	ICPUMeshDataFormatDesc* newDesc = new ICPUMeshDataFormatDesc();
	outbuffer->setMeshDataAndFormat(newDesc);
	newDesc->drop();
	outbuffer->setPrimitiveType(_inbuffer->getPrimitiveType());
	outbuffer->getMaterial() = _inbuffer->getMaterial();
	outbuffer->setBoundingBox(_inbuffer->getBoundingBox());
	outbuffer->setPositionAttributeIx(_inbuffer->getPositionAttributeIx());

	std::set<uint32_t> uniqIdxs;

	const void* const idxs = _inbuffer->getIndices();
	const video::E_INDEX_TYPE idxType =  _inbuffer->getIndexType();

	bool continuous = true; // indices are i.e. 0,1,2,3,4,5,... (also implies indices being unique)
	bool unique = true; // indices are unique (but not necessarily continuos)
	for (size_t i = 0; i < _inbuffer->getIndexCount(); ++i)
	{
		uint32_t idx = 0, prevIdx = 0xffffffff;
		if (idxType == video::EIT_16BIT)
		{
			idx = ((uint16_t*)idxs)[i];
			if (i)
				prevIdx = ((uint16_t*)idxs)[i-1];
		}
		else if (idxType == video::EIT_32BIT)
		{
			idx = ((uint32_t*)idxs)[i];
			if (i)
				prevIdx = ((uint32_t*)idxs)[i-1];
		}

		if (idx != prevIdx+1)
			continuous = false;

		unique = !uniqIdxs.insert(idx).second ? false : unique;
	}

	core::ICPUBuffer* newIdxBuffer = NULL;
	bool verticesMustBeReordered = false;
	bool needToRunForsyth = false;

	if (!continuous)
	{
		if (unique)
		{
			// no index buffer
			// vertices have to be reordered
			verticesMustBeReordered = true;
		}
		else
		{
			const uint32_t minIdx = *uniqIdxs.begin();
			video::E_INDEX_TYPE newIdxType = video::EIT_32BIT;

			if (*uniqIdxs.rbegin() - minIdx <= USHRT_MAX)
				newIdxType = video::EIT_16BIT;

			newIdxBuffer = new core::ICPUBuffer(_inbuffer->getIndexCount() * (newIdxType == video::EIT_16BIT ? 2 : 4));
			outbuffer->setIndexType(newIdxType);
			outbuffer->setIndexCount(_inbuffer->getIndexCount());
			outbuffer->setBaseVertex(minIdx); // needed???

			if (idxType == newIdxType)
				memcpy(newIdxBuffer->getPointer(), idxs, newIdxBuffer->getSize());
			else
			{
				for (size_t i = 0; i < outbuffer->getIndexCount(); ++i)
					((uint16_t*)newIdxBuffer->getPointer())[i] = ((uint32_t*)idxs)[i] - minIdx;
			}

			if (outbuffer->getPrimitiveType() == EPT_TRIANGLES)
			{
				needToRunForsyth = true;
			}
		}
	}
	// else -- no index buffer and no need to do anything

	if (newIdxBuffer)
	{
		outbuffer->getMeshDataAndFormat()->mapIndexBuffer(newIdxBuffer);
		newIdxBuffer->drop();
	}

	Attrib newAttribs[EVAI_COUNT];
	for (size_t i = 0; i < EVAI_COUNT; ++i)
		newAttribs[i].vaid = (E_VERTEX_ATTRIBUTE_ID)i;
	std::map<E_VERTEX_ATTRIBUTE_ID, std::vector<SIntegerAttr>> attribsI;
	std::map<E_VERTEX_ATTRIBUTE_ID, std::vector<core::vectorSIMDf>> attribsF;
	for (size_t vaid = EVAI_ATTR0; vaid < (size_t)EVAI_COUNT; ++vaid)
	{
		const E_COMPONENT_TYPE type = _inbuffer->getMeshDataAndFormat()->getAttribType((E_VERTEX_ATTRIBUTE_ID)vaid);

		if (_inbuffer->getMeshDataAndFormat()->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)vaid))
		{
			if (!scene::isNormalized(type) && (scene::isNativeInteger(type) || scene::isWeakInteger(type)))
				attribsI[(E_VERTEX_ATTRIBUTE_ID)vaid] = findBetterFormatI(&newAttribs[vaid].type, &newAttribs[vaid].size, &newAttribs[vaid].cpa, _inbuffer, (E_VERTEX_ATTRIBUTE_ID)vaid);
			else
				attribsF[(E_VERTEX_ATTRIBUTE_ID)vaid] = findBetterFormatF(&newAttribs[vaid].type, &newAttribs[vaid].size, &newAttribs[vaid].cpa, _inbuffer, (E_VERTEX_ATTRIBUTE_ID)vaid);
		}
	}

	size_t vertexSize = 0;
	for (size_t i = 0; i < EVAI_COUNT; ++i)
		vertexSize += newAttribs[i].size;
	// round up vertexSize to some val
	// ((X+(MUL-1))/MUL)*MUL
#ifdef _DEBUG
	{
		std::set<size_t> sizesSet;
		for (std::map<E_VERTEX_ATTRIBUTE_ID, std::vector<SIntegerAttr>>::iterator it = attribsI.begin(); it != attribsI.end(); ++it)
			sizesSet.insert(it->second.size());
		for (std::map<E_VERTEX_ATTRIBUTE_ID, std::vector<core::vectorSIMDf>>::iterator it = attribsF.begin(); it != attribsF.end(); ++it)
			sizesSet.insert(it->second.size());
		_IRR_DEBUG_BREAK_IF(sizesSet.size() != 1);
	}
#endif
	const size_t vertexCnt = (!attribsI.empty() ? attribsI.begin()->second.size() : (!attribsF.empty() ? attribsF.begin()->second.size() : 0));

	if (needToRunForsyth)
	{
		CForsythVertexCacheOptimizer forsyth;
		void* indices = newIdxBuffer->getPointer();
		if (outbuffer->getIndexType() == video::EIT_16BIT)
			forsyth.optimizeTriangleOrdering<uint16_t>(vertexCnt, outbuffer->getIndexCount(), (uint16_t*)indices, (uint16_t*)indices);
		else
			forsyth.optimizeTriangleOrdering<uint32_t>(vertexCnt, outbuffer->getIndexCount(), (uint32_t*)indices, (uint32_t*)indices);
	}

	core::ICPUBuffer* newVertexBuffer = new core::ICPUBuffer(vertexCnt * vertexSize);

	std::sort(newAttribs, newAttribs+EVAI_COUNT, std::greater<Attrib>());
	for (size_t i = 0; i < EVAI_COUNT; ++i)
	{
		newAttribs[i].offset = (i ? newAttribs[i-1].offset + newAttribs[i-1].size : 0);
		if (newAttribs[i].size)
			newDesc->mapVertexAttrBuffer(newVertexBuffer, newAttribs[i].vaid, newAttribs[i].cpa, newAttribs[i].type, vertexSize, newAttribs[i].offset);

		std::map<E_VERTEX_ATTRIBUTE_ID, std::vector<SIntegerAttr>>::iterator iti = attribsI.find((E_VERTEX_ATTRIBUTE_ID)i);
		if (iti != attribsI.end())
		{
			std::vector<SIntegerAttr>& a = iti->second;
			for (size_t ai = 0; ai < a.size(); ++ai)
				outbuffer->setAttribute(a[ai].pointer, (E_VERTEX_ATTRIBUTE_ID)i, ai);
			continue;
		}

		std::map<E_VERTEX_ATTRIBUTE_ID, std::vector<core::vectorSIMDf>>::iterator itf = attribsF.find((E_VERTEX_ATTRIBUTE_ID)i);
		if (itf != attribsF.end())
		{
			std::vector<core::vectorSIMDf>& a = itf->second;
			for (size_t ai = 0; ai < a.size(); ++ai)
				outbuffer->setAttribute(a[ai], (E_VERTEX_ATTRIBUTE_ID)i, ai);
		}
	}
	newVertexBuffer->drop();

	if (verticesMustBeReordered)
	{
		// reorder vertices according to OLD index buffer
#define ACCESS_IDX(n) ((idxType == video::EIT_32BIT) ? *((uint32_t*)(idxs)+n) : *((uint16_t*)(idxs)+n))
		uint8_t* v = (uint8_t*)newVertexBuffer->getPointer();
		for (size_t s = 1, d; s < outbuffer->getIndexCount(); ++s)
		{
			for (d = ACCESS_IDX(s); d < s; d = ACCESS_IDX(d)) ;
			if (d == s)
			{
				while (d = ACCESS_IDX(d), d != s)
				{
					uint8_t tmp[1000];
					_IRR_DEBUG_BREAK_IF(vertexSize > sizeof(tmp));
					memcpy(tmp, v + (vertexSize*s), vertexSize);
					memcpy(v + (vertexSize*s), v + (vertexSize*d), vertexSize);
					memcpy(v + (vertexSize*d), tmp, vertexSize);
				}
			}
		}
#undef ACCESS_IDX
	}

	return outbuffer;
}

std::vector<core::vectorSIMDf> CMeshManipulator::findBetterFormatF(E_COMPONENT_TYPE* _outType, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const ICPUMeshBuffer* _meshbuffer, E_VERTEX_ATTRIBUTE_ID _attrId) const
{
	const E_COMPONENT_TYPE suppTypes[]
	{
		ECT_FLOAT,
		ECT_HALF_FLOAT,
		ECT_DOUBLE_IN_FLOAT_OUT,
		ECT_UNSIGNED_INT_10F_11F_11F_REV,
		ECT_DOUBLE_IN_DOUBLE_OUT,
		ECT_NORMALIZED_INT_2_10_10_10_REV,
		ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV,
		ECT_NORMALIZED_BYTE,
		ECT_NORMALIZED_UNSIGNED_BYTE,
		ECT_NORMALIZED_SHORT,
		ECT_NORMALIZED_UNSIGNED_SHORT,
		ECT_NORMALIZED_INT,
		ECT_NORMALIZED_UNSIGNED_INT
	};

	const E_COMPONENT_TYPE thisType = _meshbuffer->getMeshDataAndFormat()->getAttribType(_attrId);
	{
		bool ok = false;
		for (size_t i = 0; i < sizeof(suppTypes)/sizeof(*suppTypes); ++i)
		{
			if (suppTypes[i] == thisType)
			{
				ok = true;
				break;
			}
		}
		if (!ok)
			return std::vector<core::vectorSIMDf>();
	}

	std::vector<core::vectorSIMDf> attribs;

	if (!_meshbuffer->getMeshDataAndFormat())
		return attribs;

	E_COMPONENTS_PER_ATTRIBUTE cpa = _meshbuffer->getMeshDataAndFormat()->getAttribComponentCount(_attrId);
	if (cpa == ECPA_REVERSED_OR_BGRA)
		cpa = ECPA_FOUR;

	float min[4]{ FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
	float max[4]{ -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

	core::vectorSIMDf attr;
	size_t idx = 0;
	while (_meshbuffer->getAttribute(attr, _attrId, idx++)) // getAttribute returns false when idx goes out of buffer's range
	{
		attribs.push_back(attr);
		for (size_t i = 0; i < cpa; ++i)
		{
			if (attr.pointer[i] < min[i])
				min[i] = attr.pointer[i];
			if (attr.pointer[i] > max[i])
				max[i] = attr.pointer[i];
		}
	}

	*_outType = getBestTypeF(scene::isNormalized(thisType), cpa, _outSize, _outCpa, min, max);
	return attribs;
}

std::vector<CMeshManipulator::SIntegerAttr> CMeshManipulator::findBetterFormatI(E_COMPONENT_TYPE* _outType, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const ICPUMeshBuffer* _meshbuffer, E_VERTEX_ATTRIBUTE_ID _attrId) const
{
	const E_COMPONENT_TYPE suppTypes[]
	{
		ECT_INT_2_10_10_10_REV,
		ECT_UNSIGNED_INT_2_10_10_10_REV,
		ECT_BYTE,
		ECT_UNSIGNED_BYTE,
		ECT_SHORT,
		ECT_UNSIGNED_SHORT,
		ECT_INT,
		ECT_UNSIGNED_INT,
		ECT_INTEGER_INT_2_10_10_10_REV,
		ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV,
		ECT_INTEGER_BYTE,
		ECT_INTEGER_UNSIGNED_BYTE,
		ECT_INTEGER_SHORT,
		ECT_INTEGER_UNSIGNED_SHORT,
		ECT_INTEGER_INT,
		ECT_INTEGER_UNSIGNED_INT
	};

	const E_COMPONENT_TYPE thisType = _meshbuffer->getMeshDataAndFormat()->getAttribType(_attrId);
	{
		bool ok = false;
		for (size_t i = 0; i < sizeof(suppTypes)/sizeof(*suppTypes); ++i)
		{
			if (suppTypes[i] == thisType)
			{
				ok = true;
				break;
			}
		}
		if (!ok)
			return std::vector<SIntegerAttr>();
	}

	std::vector<SIntegerAttr> attribs;

	if (!_meshbuffer->getMeshDataAndFormat())
		return attribs;

	E_COMPONENTS_PER_ATTRIBUTE cpa = _meshbuffer->getMeshDataAndFormat()->getAttribComponentCount(_attrId);
	if (cpa == ECPA_REVERSED_OR_BGRA)
		cpa = ECPA_FOUR;

	uint32_t min[4];
	uint32_t max[4];
	if (isUnsigned(thisType))
		for (size_t i = 0; i < 4; ++i)
			min[i] = UINT_MAX;
	else
		for (size_t i = 0; i < 4; ++i)
			min[i] = INT_MAX;
	if (isUnsigned(thisType))
		for (size_t i = 0; i < 4; ++i)
			max[i] = 0;
	else
		for (size_t i = 0; i < 4; ++i)
			max[i] = INT_MIN;

	SIntegerAttr attr;
	size_t idx = 0;
	while (_meshbuffer->getAttribute(attr.pointer, _attrId, idx++)) // getAttribute returns false when idx goes out of buffer's range
	{
		attribs.push_back(attr);
		for (size_t i = 0; i < cpa; ++i)
		{
			if (scene::isUnsigned(thisType))
			{
				if (attr.pointer[i] < min[i])
					min[i] = attr.pointer[i];
				if (attr.pointer[i] > max[i])
					max[i] = attr.pointer[i];
			}
			else
			{
				if (((int32_t*)attr.pointer + i)[0] < ((int32_t*)min + i)[0])
					min[i] = attr.pointer[i];
				if (((int32_t*)attr.pointer + i)[0] > ((int32_t*)max + i)[0])
					max[i] = attr.pointer[i];
			}
		}
	}

	*_outType = getBestTypeI(scene::isNativeInteger(thisType), scene::isUnsigned(thisType), cpa, _outSize, _outCpa, min, max);
	return attribs;
}

E_COMPONENT_TYPE CMeshManipulator::getBestTypeF(bool _normalized, E_COMPONENTS_PER_ATTRIBUTE _cpa, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const float* _min, const float* _max) const
{
	std::set<E_COMPONENT_TYPE> all;
	{
		E_COMPONENT_TYPE arrayAll[]{ ECT_FLOAT, ECT_HALF_FLOAT, ECT_DOUBLE_IN_FLOAT_OUT, ECT_DOUBLE_IN_DOUBLE_OUT, ECT_UNSIGNED_INT_10F_11F_11F_REV, ECT_NORMALIZED_INT_2_10_10_10_REV, ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV, ECT_NORMALIZED_BYTE, ECT_NORMALIZED_UNSIGNED_BYTE, ECT_NORMALIZED_SHORT, ECT_NORMALIZED_UNSIGNED_SHORT, ECT_NORMALIZED_INT, ECT_NORMALIZED_UNSIGNED_INT};
		for (size_t i = 0; i < sizeof(arrayAll)/sizeof(*arrayAll); ++i)
			all.insert(arrayAll[i]);
	}
	std::set<E_COMPONENT_TYPE> normalized;
	{
		E_COMPONENT_TYPE arrayNormalized[]{ ECT_NORMALIZED_INT_2_10_10_10_REV, ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV, ECT_NORMALIZED_BYTE, ECT_NORMALIZED_UNSIGNED_BYTE, ECT_NORMALIZED_SHORT, ECT_NORMALIZED_UNSIGNED_SHORT, ECT_NORMALIZED_INT, ECT_NORMALIZED_UNSIGNED_INT };
		for (size_t i = 0; i < sizeof(arrayNormalized)/sizeof(*arrayNormalized); ++i)
			normalized.insert(arrayNormalized[i]);
	}

	if (_normalized)
		all = normalized;
	else
	{
		for (std::set<E_COMPONENT_TYPE>::iterator it = normalized.begin(); it != normalized.end(); ++it)
			all.erase(*it);
	}

	E_COMPONENT_TYPE bestType = _normalized ? ECT_NORMALIZED_INT : ECT_DOUBLE_IN_DOUBLE_OUT;
	for (std::set<E_COMPONENT_TYPE>::iterator it = all.begin(); it != all.end(); ++it)
	{
		bool validComb = false;
		E_COMPONENTS_PER_ATTRIBUTE chosenCpa = _cpa; // find cpa compatible with currently considered type
		for (size_t c = _cpa; c <= ECPA_FOUR; ++c)
		{
			if (validCombination(*it, (E_COMPONENTS_PER_ATTRIBUTE)c))
			{
				chosenCpa = (E_COMPONENTS_PER_ATTRIBUTE)c;
				validComb = true;
				break;
			}
		}
		if (validComb)
		{
			bool ok = true;
			for (size_t cmpntNum = 0; cmpntNum < _cpa; ++cmpntNum) // check only `_cpa` components because even if (chosenCpa > _cpa), we don't care about extra components
			{
				if (!(_min[cmpntNum] >= minValueOfTypeFP(*it, cmpntNum) && _max[cmpntNum] <= maxValueOfTypeFP(*it, cmpntNum)))
				{
					ok = false;
					break;
				}
			}
			if (ok && vertexAttrSize[*it][chosenCpa] < vertexAttrSize[bestType][chosenCpa]) // vertexAttrSize array defined in IMeshBuffer.h
			{
				bestType = *it;
				*_outSize = vertexAttrSize[bestType][chosenCpa];
				*_outCpa = chosenCpa;
			}
		}
	}
	return bestType;
}

E_COMPONENT_TYPE CMeshManipulator::getBestTypeI(bool _nativeInt, bool _unsigned, E_COMPONENTS_PER_ATTRIBUTE _cpa, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const uint32_t* _min, const uint32_t* _max) const
{
	std::set<E_COMPONENT_TYPE> all;
	{
		E_COMPONENT_TYPE arrayAll[]{ ECT_INT_2_10_10_10_REV, ECT_UNSIGNED_INT_2_10_10_10_REV, ECT_BYTE, ECT_UNSIGNED_BYTE, ECT_SHORT, ECT_UNSIGNED_SHORT, ECT_INT, ECT_UNSIGNED_INT, /*ECT_INTEGER_INT_2_10_10_10_REV, ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV,*/ ECT_INTEGER_BYTE, ECT_INTEGER_UNSIGNED_BYTE, ECT_INTEGER_SHORT, ECT_INTEGER_UNSIGNED_SHORT, ECT_INTEGER_INT, ECT_INTEGER_UNSIGNED_INT };
		for (size_t i = 0; i < sizeof(arrayAll)/sizeof(*arrayAll); ++i)
			all.insert(arrayAll[i]);
	}
	std::set<E_COMPONENT_TYPE> nativeInts;
	{
		E_COMPONENT_TYPE arrayNative[]{ ECT_INTEGER_INT_2_10_10_10_REV, ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV, ECT_INTEGER_BYTE, ECT_INTEGER_UNSIGNED_BYTE, ECT_INTEGER_SHORT, ECT_INTEGER_UNSIGNED_SHORT, ECT_INTEGER_INT, ECT_INTEGER_UNSIGNED_INT };
		for (size_t i = 0; i < sizeof(arrayNative)/sizeof(*arrayNative); ++i)
			nativeInts.insert(arrayNative[i]);
	}

	if (_nativeInt)
		all = nativeInts;
	else
	{
		for (std::set<E_COMPONENT_TYPE>::iterator it = nativeInts.begin(); it != nativeInts.end(); ++it)
			all.erase(*it);
	}

	E_COMPONENT_TYPE bestType = _nativeInt ? (_unsigned ? ECT_INTEGER_UNSIGNED_INT : ECT_INTEGER_INT) : (_unsigned ? ECT_UNSIGNED_INT : ECT_INT);
	for (std::set<E_COMPONENT_TYPE>::iterator it = all.begin(); it != all.end(); ++it)
	{
		bool validComb = false;
		E_COMPONENTS_PER_ATTRIBUTE chosenCpa = _cpa; // cpa compatible with currently considered type
		for (size_t c = _cpa; c <= ECPA_FOUR; ++c)
		{
			if (validCombination(*it, (E_COMPONENTS_PER_ATTRIBUTE)c))
			{
				chosenCpa = (E_COMPONENTS_PER_ATTRIBUTE)c;
				validComb = true;
				break;
			}
		}
		if (validComb)
		{
			bool ok = true;
			for (size_t cmpntNum = 0; cmpntNum < _cpa; ++cmpntNum) // check only `_cpa` components because even if (chosenCpa > _cpa), we don't care about extra components
			{
				if (_unsigned)
				{
					if (!(_min[cmpntNum] >= minValueOfTypeINT(*it, cmpntNum) && _max[cmpntNum] <= maxValueOfTypeINT(*it, cmpntNum)))
					{
						ok = false;
						break;
					}
				}
				else
				{
					if (!(((int32_t*)(_min + cmpntNum))[0] >= minValueOfTypeINT(*it, cmpntNum) && ((int32_t*)(_max + cmpntNum))[0] <= maxValueOfTypeINT(*it, cmpntNum)))
					{
						ok = false;
						break;
					}
				}
			}
			if (ok && vertexAttrSize[*it][chosenCpa] < vertexAttrSize[bestType][chosenCpa]) // vertexAttrSize array defined in IMeshBuffer.h
			{
				bestType = *it;
				*_outSize = vertexAttrSize[bestType][chosenCpa];
				*_outCpa = chosenCpa;
			}
		}
	}
	return bestType;
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
	for (uint32_t g=0; g<mesh->getMeshBufferCount(); ++g)
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
	core::array<uint32_t> tris;
	float score;
	int16_t cachepos;
	uint16_t NumActiveTris;
};

struct tcache
{
	uint16_t ind[3];
	float score;
	bool drawn;
};

const uint16_t cachesize = 32;

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
		for (uint16_t i = 0; i < cachesize; i++)
		{
			cache[i] = -1;
		}
	}

	// Adds this vertex index and returns the highest-scoring triangle index
	uint32_t add(uint16_t vert, bool updatetris = false)
	{
		bool found = false;

		// Mark existing pos as empty
		for (uint16_t i = 0; i < cachesize; i++)
		{
			if (cache[i] == vert)
			{
				// Move everything down
				for (uint16_t j = i; j; j--)
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
			for (uint16_t i = cachesize - 1; i; i--)
			{
				cache[i] = cache[i - 1];
			}
		}

		cache[0] = vert;

		uint32_t highest = 0;
		float hiscore = 0;

		if (updatetris)
		{
			// Update cache positions
			for (uint16_t i = 0; i < cachesize; i++)
			{
				if (cache[i] == -1)
					break;

				vc[cache[i]].cachepos = i;
				vc[cache[i]].score = FindVertexScore(&vc[cache[i]]);
			}

			// Update triangle scores
			for (uint16_t i = 0; i < cachesize; i++)
			{
				if (cache[i] == -1)
					break;

				const uint16_t trisize = vc[cache[i]].tris.size();
				for (uint16_t t = 0; t < trisize; t++)
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
	int32_t cache[cachesize];
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

	const uint32_t mbcount = mesh->getMeshBufferCount();

	for (uint32_t b = 0; b < mbcount; ++b)
	{
		const IMeshBuffer *mb = mesh->getMeshBuffer(b);

		if (mb->getIndexType() != video::EIT_16BIT)
		{
			os::Printer::log("Cannot optimize a mesh with 32bit indices", ELL_ERROR);
			newmesh->drop();
			return 0;
		}

		const uint32_t icount = mb->getIndexCount();
		const uint32_t tcount = icount / 3;
		const uint32_t vcount = mb->getVertexCount();
		const uint16_t* ind = reinterpret_cast<const uint16_t*>(mb->getIndices());

		vcache *vc = new vcache[vcount];
		tcache *tc = new tcache[tcount];

		f_lru lru(vc, tc);

		// init
		for (uint16_t i = 0; i < vcount; i++)
		{
			vc[i].score = 0;
			vc[i].cachepos = -1;
			vc[i].NumActiveTris = 0;
		}

		// First pass: count how many times a vert is used
		for (uint32_t i = 0; i < icount; i += 3)
		{
			vc[ind[i]].NumActiveTris++;
			vc[ind[i + 1]].NumActiveTris++;
			vc[ind[i + 2]].NumActiveTris++;

			const uint32_t tri_ind = i/3;
			tc[tri_ind].ind[0] = ind[i];
			tc[tri_ind].ind[1] = ind[i + 1];
			tc[tri_ind].ind[2] = ind[i + 2];
		}

		// Second pass: list of each triangle
		for (uint32_t i = 0; i < tcount; i++)
		{
			vc[tc[i].ind[0]].tris.push_back(i);
			vc[tc[i].ind[1]].tris.push_back(i);
			vc[tc[i].ind[2]].tris.push_back(i);

			tc[i].drawn = false;
		}

		// Give initial scores
		for (uint16_t i = 0; i < vcount; i++)
		{
			vc[i].score = FindVertexScore(&vc[i]);
		}
		for (uint32_t i = 0; i < tcount; i++)
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

				core::map<const video::S3DVertex, const uint16_t> sind; // search index for fast operation
				typedef core::map<const video::S3DVertex, const uint16_t>::Node snode;

				// Main algorithm
				uint32_t highest = 0;
				uint32_t drawcalls = 0;
				for (;;)
				{
					if (tc[highest].drawn)
					{
						bool found = false;
						float hiscore = 0;
						for (uint32_t t = 0; t < tcount; t++)
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
					uint16_t newind = buf->Vertices.size();

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

					for (uint16_t j = 0; j < 3; j++)
					{
						vcache *vert = &vc[tc[highest].ind[j]];
						for (uint16_t t = 0; t < vert->tris.size(); t++)
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

				core::map<const video::S3DVertex2TCoords, const uint16_t> sind; // search index for fast operation
				typedef core::map<const video::S3DVertex2TCoords, const uint16_t>::Node snode;

				// Main algorithm
				uint32_t highest = 0;
				uint32_t drawcalls = 0;
				for (;;)
				{
					if (tc[highest].drawn)
					{
						bool found = false;
						float hiscore = 0;
						for (uint32_t t = 0; t < tcount; t++)
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
					uint16_t newind = buf->Vertices.size();

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

					for (uint16_t j = 0; j < 3; j++)
					{
						vcache *vert = &vc[tc[highest].ind[j]];
						for (uint16_t t = 0; t < vert->tris.size(); t++)
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

				core::map<const video::S3DVertexTangents, const uint16_t> sind; // search index for fast operation
				typedef core::map<const video::S3DVertexTangents, const uint16_t>::Node snode;

				// Main algorithm
				uint32_t highest = 0;
				uint32_t drawcalls = 0;
				for (;;)
				{
					if (tc[highest].drawn)
					{
						bool found = false;
						float hiscore = 0;
						for (uint32_t t = 0; t < tcount; t++)
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
					uint16_t newind = buf->Vertices.size();

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

					for (uint16_t j = 0; j < 3; j++)
					{
						vcache *vert = &vc[tc[highest].ind[j]];
						for (uint16_t t = 0; t < vert->tris.size(); t++)
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

