// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_S_TRIANGLE_MESH_COLLIDER_H_INCLUDED__
#define __NBL_S_TRIANGLE_MESH_COLLIDER_H_INCLUDED__

#include "SAABoxCollider.h"
#include "nbl/core/IReferenceCounted.h"

namespace nbl
{
namespace core
{
class STriangleCollider  // : public AllocationOverrideDefault EBO inheritance problem
{
public:
    STriangleCollider() {}
    STriangleCollider(const vectorSIMDf& A, const vectorSIMDf& B, const vectorSIMDf& C, bool& validTriangle)
    {
        const vectorSIMDf edges[2] = {B - A, C - A};

        vectorSIMDf normal = planeEq = cross(edges[0], edges[1]);
        if((normal == vectorSIMDf(0.f)).all())
        {
            validTriangle = false;
            return;
        }
        boundaryPlanes[0] = cross(normal, edges[0]);
        boundaryPlanes[1] = cross(edges[1], normal);
        boundaryPlanes[0] /= dot(boundaryPlanes[0], edges[1]);
        boundaryPlanes[1] /= dot(boundaryPlanes[1], edges[0]);

        planeEq.W = dot(planeEq, A).X;
        boundaryPlanes[0].W = -dot(boundaryPlanes[0], A).X;
        boundaryPlanes[1].W = -dot(boundaryPlanes[1], A).X;
        validTriangle = true;
    }

    inline bool CollideWithRay(float& collisionDistance, vectorSIMDf origin, vectorSIMDf direction, const float& dirMaxMultiplier) const
    {
        direction.makeSafe3D();
        origin.makeSafe3D();

        const float NdotD = dot(direction, planeEq).X;
        if(NdotD != 0.f)
            return false;

        const float NdotOrigin = dot(origin, planeEq).X;
        const float d = planeEq.W;

        const float t = (d - NdotOrigin) / NdotD;
        if(t >= dirMaxMultiplier || t < 0.f)
            return false;

        vectorSIMDf outPoint = origin + direction * t;

        vectorSIMDf extraComponent(0.f, 0.f, 0.f, 1.f);
        const vectorSIMDf outPointW1 = outPoint | reinterpret_cast<const vectorSIMDu32&>(extraComponent);

        const float distToEdge[2] = {dot(outPointW1, boundaryPlanes[0])[0], dot(outPointW1, boundaryPlanes[1])[0]};
        if(distToEdge[0] < 0.f || distToEdge[1] < 0.f || (distToEdge[0] + distToEdge[1]) > 1.f)
        {
            collisionDistance = t;
            return true;
        }
        else
            return false;
    }

    vectorSIMDf planeEq;
    vectorSIMDf boundaryPlanes[2];
};

class STriangleMeshCollider : public IReferenceCounted
{
    _NBL_INTERFACE_CHILD(STriangleMeshCollider) {}

    SAABoxCollider BBox;
    ///matrix4x3 cachedTransformInverse;
    ///matrix4x3 cachedTransform;
    vector<STriangleCollider> triangles;

public:
    STriangleMeshCollider()
        : BBox(core::aabbox3df()) {}

    inline const SAABoxCollider& getBoundingBox() const { return BBox; }

    inline size_t getTriangleCount() const { return triangles.size(); }

    inline bool Init(float* vertices, const size_t& indexCount, uint32_t* indices = NULL)
    {
        bool firstPoint = true;
        if(indices)
        {
            for(size_t i = 0; i < indexCount; i += 3)
            {
                vectorSIMDf A(vertices[indices[i + 0] * 3 + 0], vertices[indices[i + 0] * 3 + 1], vertices[indices[i + 0] * 3 + 2]);
                vectorSIMDf B(vertices[indices[i + 1] * 3 + 0], vertices[indices[i + 1] * 3 + 1], vertices[indices[i + 1] * 3 + 2]);
                vectorSIMDf C(vertices[indices[i + 2] * 3 + 0], vertices[indices[i + 2] * 3 + 1], vertices[indices[i + 2] * 3 + 2]);

                bool useful = false;
                STriangleCollider triangle(A, B, C, useful);
                if(useful)
                {
                    if(firstPoint)
                    {
                        BBox.Box.reset(A.getAsVector3df());
                        firstPoint = false;
                    }
                    else
                        BBox.Box.addInternalPoint(A.getAsVector3df());
                    BBox.Box.addInternalPoint(B.getAsVector3df());
                    BBox.Box.addInternalPoint(C.getAsVector3df());
                    triangles.push_back(triangle);
                }
            }
        }
        else
        {
            for(size_t i = 0; i < indexCount; i += 3)
            {
                vectorSIMDf A(vertices[(i + 0) * 3 + 0], vertices[(i + 0) * 3 + 1], vertices[(i + 0) * 3 + 2]);
                vectorSIMDf B(vertices[(i + 1) * 3 + 0], vertices[(i + 1) * 3 + 1], vertices[(i + 1) * 3 + 2]);
                vectorSIMDf C(vertices[(i + 2) * 3 + 0], vertices[(i + 2) * 3 + 1], vertices[(i + 2) * 3 + 2]);

                bool useful;
                STriangleCollider triangle(A, B, C, useful);
                if(useful)
                {
                    if(firstPoint)
                    {
                        BBox.Box.reset(A.getAsVector3df());
                        firstPoint = false;
                    }
                    else
                        BBox.Box.addInternalPoint(A.getAsVector3df());
                    BBox.Box.addInternalPoint(B.getAsVector3df());
                    BBox.Box.addInternalPoint(C.getAsVector3df());
                    triangles.push_back(triangle);
                }
            }
        }

        return triangles.size();
    }

    inline bool CollideWithRay(float& collisionDistance, const vectorSIMDf& origin, const vectorSIMDf& direction, const float& dirMaxMultiplier) const
    {
        return CollideWithRay(collisionDistance, origin, direction, dirMaxMultiplier, reciprocal_approxim(direction));
    }

    inline bool CollideWithRay(float& collisionDistance, const vectorSIMDf& origin, const vectorSIMDf& direction, const float& dirMaxMultiplier, const vectorSIMDf& direction_reciprocal) const
    {
        float dummyDist;
        if(!BBox.CollideWithRay(dummyDist, origin, direction, dirMaxMultiplier, direction_reciprocal))
            return false;

        for(size_t i = 0; i < triangles.size(); i++)
        {
            if(triangles[i].CollideWithRay(collisionDistance, origin, direction, dirMaxMultiplier))
                return true;
        }

        return false;
    }
    /**
        inline bool UpdateTransformation(const matrix4x3& newTransform)
        {
            if (newTransform==cachedTransform)
                return false;

            matrix4x3 newInverse;
            if (!newTransform.getInverse(newInverse))
                return false;

            matrix4x3 diffMatrix = newTransform*cachedTransformInverse;0AZ8I
            diffMatrix[4] = 0.f;
            diffMatrix[8] = 0.f;
            diffMatrix[12] = 0.f;
            diffMatrix[13] = 0.f;
            diffMatrix[14] = 0.f;
            diffMatrix[15] = 1.f;
            cachedTransform = newTransform;
            cachedTransformInverse = newInverse;

            matrix4x3 diffMatrix_NormalMatrix;
            diffMatrix.getInverse(diffMatrix_NormalMatrix);
            diffMatrix_NormalMatrix = diffMatrix_NormalMatrix.getTransposed();

            for (size_t i=0; i<triangles.size(); i++)
            {
                matrix4x3 T_Old;
                T_Old[0] = triangles[i].planeEq[0];
                T_Old[4] = triangles[i].planeEq[1];
                T_Old[8] = triangles[i].planeEq[2];
                T_Old[1] = triangles[i].boundaryPlanes[0][0];
                T_Old[5] = triangles[i].boundaryPlanes[0][1];
                T_Old[9] = triangles[i].boundaryPlanes[0][2];
                T_Old[2] = triangles[i].boundaryPlanes[1][0];
                T_Old[6] = triangles[i].boundaryPlanes[1][1];
                T_Old[10] = triangles[i].boundaryPlanes[1][2];
                T_Old.makeInverse();

                float D[3];
                D[0] = triangles.planeEq[3];
                D[1] = triangles.boundaryPlanes[0][3];
                D[2] = triangles.boundaryPlanes[1][3];
                vector3df A;
                T_Old.rotateVect(A,(vector3df*)D);///this is actually multiply with 3x3 submatrix

                diffMatrix_NormalMatrix(triangles[i].planeEq.getAsVector3df());
                diffMatrix.rotateVect(triangles[i].boundaryPlanes[0].getAsVector3df());///this is actually multiply with 3x3 submatrix
                diffMatrix.rotateVect(triangles[i].boundaryPlanes[1].getAsVector3df());///this is actually multiply with 3x3 submatrix
                diffMatrix.transformVect(A);


                matrix4x3 T;
                T[0] = triangles[i].planeEq[0];
                T[4] = triangles[i].planeEq[1];
                T[8] = triangles[i].planeEq[2];
                T[1] = triangles[i].boundaryPlanes[0][0];
                T[5] = triangles[i].boundaryPlanes[0][1];
                T[9] = triangles[i].boundaryPlanes[0][2];
                T[2] = triangles[i].boundaryPlanes[1][0];
                T[6] = triangles[i].boundaryPlanes[1][1];
                T[10] = triangles[i].boundaryPlanes[1][2];

                T.rotateVect(A);///this is actually multiply with 3x3 submatrix
                triangles[i].planeEq[3] = A.X;
                triangles[i].boundaryPlanes[0][3] = A.Y;
                triangles[i].boundaryPlanes[1][3] = A.Z;
            }
        }**/
};

}
}

#endif
