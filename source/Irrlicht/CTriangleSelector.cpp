// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CTriangleSelector.h"
#include "ISceneNode.h"
#include "IMeshBuffer.h"

namespace irr
{
namespace scene
{

//! constructor
CTriangleSelector::CTriangleSelector(ISceneNode* node)
: SceneNode(node)
{
	#ifdef _DEBUG
	setDebugName("CTriangleSelector");
	#endif

	BoundingBox.reset(0.f, 0.f, 0.f);
}


//! constructor
CTriangleSelector::CTriangleSelector(const core::aabbox3d<f32>& box, ISceneNode* node)
: SceneNode(node)
{
	#ifdef _DEBUG
	setDebugName("CTriangleSelector");
	#endif

	BoundingBox=box;
	// TODO
}


//! constructor
CTriangleSelector::CTriangleSelector(const ICPUMesh* mesh, ISceneNode* node)
: SceneNode(node)
{
	#ifdef _DEBUG
	setDebugName("CTriangleSelector");
	#endif

	createFromMesh(mesh);
}


/*

        switch (mesh->getMeshBuffer(j))
        {
            case EPT_TRIANGLES:
                break;
            case EPT_TRIANGLE_FAN:
                break;
            case EPT_TRIANGLE_STRIP:
                break;
        }
*/
template<class I>
inline void updateTriangles(const ICPUMeshBuffer* buf, core::aabbox3df &BoundingBox, core::array<core::triangle3df> &Triangles, u32 &triangleCount)
{
    const uint64_t idxCnt = buf->getIndexCount();
    if (idxCnt<3)
    {
        BoundingBox.reset(0,0,0);
        return;
    }
    const I* const indices = reinterpret_cast<const I*>(buf->getIndices());

    //
    u32 firstTrianglePlus1 = triangleCount+1;
    switch (buf->getPrimitiveType())
    {
        case EPT_TRIANGLES:
            for (uint64_t j=0; j<idxCnt; j+=3)
            {
                core::triangle3df& tri = Triangles[triangleCount++];
                if (indices)
                {
                    tri.pointA = buf->getPosition(indices[j + 0]).getAsVector3df();
                    tri.pointB = buf->getPosition(indices[j + 1]).getAsVector3df();
                    tri.pointC = buf->getPosition(indices[j + 2]).getAsVector3df();
                }
                else
                {
                    tri.pointA = buf->getPosition(j+0).getAsVector3df();
                    tri.pointB = buf->getPosition(j+1).getAsVector3df();
                    tri.pointC = buf->getPosition(j+2).getAsVector3df();
                }
                if (triangleCount!=firstTrianglePlus1)
                    BoundingBox.addInternalPoint(tri.pointA);
                else
                    BoundingBox.reset(tri.pointA);
                BoundingBox.addInternalPoint(tri.pointB);
                BoundingBox.addInternalPoint(tri.pointC);
            }
            break;
        case EPT_TRIANGLE_FAN:
            {
                core::triangle3df tri;
                I triangleLastIx;
                if (indices)
                {
                    tri.pointA = buf->getPosition(indices[0]).getAsVector3df();
                    triangleLastIx = indices[1];
                }
                else
                {
                    tri.pointA = buf->getPosition(0).getAsVector3df();
                    triangleLastIx = 1;
                }
                for (uint64_t j=2; j<idxCnt; j++)
                {
                    I currentIx;
                    if (indices)
                        currentIx = indices[j];
                    else
                        currentIx = j;
                    //
                    tri.pointB = buf->getPosition(triangleLastIx).getAsVector3df();
                    tri.pointC = buf->getPosition(currentIx).getAsVector3df();
                    triangleLastIx = currentIx;/*
                    if (tri.pointA==tri.pointB||tri.pointB==tri.pointC||tri.pointA==tri.pointC)
                        continue;
                    */
                    Triangles[triangleCount++] = tri;
                    if (triangleCount!=firstTrianglePlus1)
                        BoundingBox.addInternalPoint(tri.pointA);
                    else
                        BoundingBox.reset(tri.pointA);
                    BoundingBox.addInternalPoint(tri.pointB);
                    BoundingBox.addInternalPoint(tri.pointC);
                }
            }
            break;
        case EPT_TRIANGLE_STRIP:
            {
                I triangleIndices[3];
                if (indices)
                {
                    //triangleIndices[0] = indices[0];
                    triangleIndices[1] = indices[1];
                    //triangleIndices[2] = indices[2];
                    triangleIndices[2] = indices[0];
                }
                else
                {
                    //triangleIndices[0] = 0;
                    triangleIndices[1] = 1;
                    //triangleIndices[2] = 2;
                    triangleIndices[2] = 0;
                }
                for (uint64_t j=2; j<idxCnt; j++)
                {
                    core::triangle3df& tri = Triangles[triangleCount++];
                    triangleIndices[0] = triangleIndices[2];
                    if (indices)
                        triangleIndices[2] = indices[j];
                    else
                        triangleIndices[2] = j;
                    //
                    tri.pointA = buf->getPosition(triangleIndices[0]).getAsVector3df();
                    tri.pointB = buf->getPosition(triangleIndices[1]).getAsVector3df();
                    tri.pointC = buf->getPosition(triangleIndices[2]).getAsVector3df();/*
                    if (tri.pointA==tri.pointB||tri.pointB==tri.pointC||tri.pointA==tri.pointC)
                    {
                        triangleCount--;
                        continue;
                    }
                    */
                    if (triangleCount!=firstTrianglePlus1)
                        BoundingBox.addInternalPoint(tri.pointA);
                    else
                        BoundingBox.reset(tri.pointA);
                    BoundingBox.addInternalPoint(tri.pointB);
                    BoundingBox.addInternalPoint(tri.pointC);
                }
            }
            break;
    }
}

void CTriangleSelector::createFromMesh(const ICPUMesh* mesh)
{
	const u32 cnt = mesh->getMeshBufferCount();
	u32 totalFaceCount = 0;
	for (u32 j=0; j<cnt; ++j)
    {
        switch (mesh->getMeshBuffer(j)->getPrimitiveType())
        {
            case EPT_TRIANGLES:
                totalFaceCount += mesh->getMeshBuffer(j)->getIndexCount()/3;
                break;
            case EPT_TRIANGLE_FAN:
                totalFaceCount += (mesh->getMeshBuffer(j)->getIndexCount()-1)/2;
                break;
            case EPT_TRIANGLE_STRIP:
                totalFaceCount += (mesh->getMeshBuffer(j)->getIndexCount()-2);
                break;
        }
    }
	Triangles.reallocate(totalFaceCount);

    u32 trianglesUsed = 0;
	for (u32 i=0; i<cnt; ++i)
	{
		const ICPUMeshBuffer* buf = mesh->getMeshBuffer(i);

		if (buf->getIndexType()==video::EIT_16BIT)
            updateTriangles<u16>(buf,BoundingBox,Triangles,trianglesUsed);
		else if (buf->getIndexType()==video::EIT_32BIT)
            updateTriangles<u32>(buf,BoundingBox,Triangles,trianglesUsed);
	}
	Triangles.set_used(trianglesUsed);
}

void CTriangleSelector::updateFromMesh(const ICPUMesh* mesh) const
{
	if (!mesh)
		return;

	u32 meshBuffers = mesh->getMeshBufferCount();
	u32 triangleCount = 0;

	for (u32 i = 0; i < meshBuffers; ++i)
	{
		ICPUMeshBuffer* buf = mesh->getMeshBuffer(i);
		if (buf->getIndexType()==video::EIT_16BIT)
            updateTriangles<u16>(buf,BoundingBox,Triangles,triangleCount);
		else if (buf->getIndexType()==video::EIT_32BIT)
            updateTriangles<u32>(buf,BoundingBox,Triangles,triangleCount);
	}
}


//! Gets all triangles.
void CTriangleSelector::getTriangles(core::triangle3df* triangles,
					s32 arraySize, s32& outTriangleCount,
					const core::matrix4* transform) const
{
	u32 cnt = Triangles.size();
	if (cnt > (u32)arraySize)
		cnt = (u32)arraySize;

	core::matrix4 mat;
	if (transform)
		mat = *transform;
	if (SceneNode)
		mat *= SceneNode->getAbsoluteTransformation();

	for (u32 i=0; i<cnt; ++i)
	{
		mat.transformVect( triangles[i].pointA, Triangles[i].pointA );
		mat.transformVect( triangles[i].pointB, Triangles[i].pointB );
		mat.transformVect( triangles[i].pointC, Triangles[i].pointC );
	}

	outTriangleCount = cnt;
}


//! Gets all triangles which lie within a specific bounding box.
void CTriangleSelector::getTriangles(core::triangle3df* triangles,
					s32 arraySize, s32& outTriangleCount,
					const core::aabbox3d<f32>& box,
					const core::matrix4* transform) const
{
	core::matrix4 mat(core::matrix4::EM4CONST_NOTHING);
	core::aabbox3df tBox(box);

	if (SceneNode)
	{
		SceneNode->getAbsoluteTransformation().getInverse(mat);
		mat.transformBoxEx(tBox);
	}
	if (transform)
		mat = *transform;
	else
		mat.makeIdentity();
	if (SceneNode)
		mat *= SceneNode->getAbsoluteTransformation();

	outTriangleCount = 0;

	if (!tBox.intersectsWithBox(BoundingBox))
		return;

	s32 triangleCount = 0;
	const u32 cnt = Triangles.size();
	for (u32 i=0; i<cnt; ++i)
	{
		// This isn't an accurate test, but it's fast, and the
		// API contract doesn't guarantee complete accuracy.
		if (Triangles[i].isTotalOutsideBox(tBox))
		   continue;

		triangles[triangleCount] = Triangles[i];
		mat.transformVect(triangles[triangleCount].pointA);
		mat.transformVect(triangles[triangleCount].pointB);
		mat.transformVect(triangles[triangleCount].pointC);

		++triangleCount;

		if (triangleCount == arraySize)
			break;
	}

	outTriangleCount = triangleCount;
}


//! Gets all triangles which have or may have contact with a 3d line.
void CTriangleSelector::getTriangles(core::triangle3df* triangles,
					s32 arraySize, s32& outTriangleCount,
					const core::line3d<f32>& line,
					const core::matrix4* transform) const
{
	core::aabbox3d<f32> box(line.start);
	box.addInternalPoint(line.end);

	// TODO: Could be optimized for line a little bit more.
	getTriangles(triangles, arraySize, outTriangleCount,
				box, transform);
}


//! Returns amount of all available triangles in this selector
s32 CTriangleSelector::getTriangleCount() const
{
	return Triangles.size();
}


/* Get the number of TriangleSelectors that are part of this one.
Only useful for MetaTriangleSelector others return 1
*/
u32 CTriangleSelector::getSelectorCount() const
{
	return 1;
}


/* Get the TriangleSelector based on index based on getSelectorCount.
Only useful for MetaTriangleSelector others return 'this' or 0
*/
ITriangleSelector* CTriangleSelector::getSelector(u32 index)
{
	if (index)
		return 0;
	else
		return this;
}


/* Get the TriangleSelector based on index based on getSelectorCount.
Only useful for MetaTriangleSelector others return 'this' or 0
*/
const ITriangleSelector* CTriangleSelector::getSelector(u32 index) const
{
	if (index)
		return 0;
	else
		return this;
}

ISceneNode* CTriangleSelector::getSceneNodeForTriangle(u32 triangleIndex) const
{
    return SceneNode;
}

} // end namespace scene
} // end namespace irr
