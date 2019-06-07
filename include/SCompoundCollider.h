#ifndef __S_COMPOUND_COLLIDER_H_INCLUDED__
#define __S_COMPOUND_COLLIDER_H_INCLUDED__

#include "SAABoxCollider.h"
#include "SEllipsoidCollider.h"
#include "STriangleMeshCollider.h"
#include "quaternion.h"

namespace irr
{
namespace core
{

struct SColliderData
{
    SColliderData() : attachedNode(NULL), instanceID(0), userData(NULL) {}
    scene::ISceneNode* attachedNode; //for relative translation
    uint32_t instanceID;
    void* userData;
};

struct SCollisionShapeDef
{
    enum E_COLLISION_SHAPE_TYPE
    {
        ECST_AABOX=0,
        ECST_ELLIPSOID,
        ECST_TRIANGLE,
        ECST_TRIANGLE_MESH,
        ECST_COUNT
    };
    E_COLLISION_SHAPE_TYPE objectType;
    void* object;
};

class SCompoundCollider : public IReferenceCounted
{
    protected:
        SAABoxCollider BBox;
        vector<SCollisionShapeDef> Shapes;
        SColliderData colliderData;

		//! Destructor.
        ~SCompoundCollider()
        {
            for (size_t i=0; i<Shapes.size(); i++)
            {
                switch (Shapes[i].objectType)
                {
                    case SCollisionShapeDef::ECST_AABOX:
                        {
                            SAABoxCollider* tmp = static_cast<SAABoxCollider*>(Shapes[i].object);
                            delete tmp;
                        }
                        break;
                    case SCollisionShapeDef::ECST_ELLIPSOID:
                        {
                            SEllipsoidCollider* tmp = static_cast<SEllipsoidCollider*>(Shapes[i].object);
                            delete tmp;
                        }
                        break;
                    case SCollisionShapeDef::ECST_TRIANGLE:
                        {
                            STriangleCollider* tmp = static_cast<STriangleCollider*>(Shapes[i].object);
                            delete tmp;
                        }
                        break;
                    case SCollisionShapeDef::ECST_TRIANGLE_MESH:
                        {
                            STriangleMeshCollider* tmp = static_cast<STriangleMeshCollider*>(Shapes[i].object);
                            tmp->drop();
                        }
                        break;
                    case SCollisionShapeDef::ECST_COUNT:
                        assert(0);
                        break;
                }
            }
        }
    public:
		//! Default constructor.
        SCompoundCollider() : BBox(aabbox3df()) {}


		//! @returns Pointer to brand new copy of `this` collider. The copy object is allocated with `new`.
        inline SCompoundCollider* clone()
        {
            SCompoundCollider* coll = new SCompoundCollider();
            coll->BBox = BBox;
            coll->colliderData = colliderData;
            for (size_t i=0; i<Shapes.size(); i++)
            {
                switch (Shapes[i].objectType)
                {
                    case SCollisionShapeDef::ECST_AABOX:
                        {
                            SAABoxCollider* tmp = static_cast<SAABoxCollider*>(Shapes[i].object);
                            coll->AddBox(*tmp);
                        }
                        break;
                    case SCollisionShapeDef::ECST_ELLIPSOID:
                        {
                            SEllipsoidCollider* tmp = static_cast<SEllipsoidCollider*>(Shapes[i].object);
                            SCollisionShapeDef shape;
                            shape.object = new SEllipsoidCollider(*tmp);
                            shape.objectType = SCollisionShapeDef::ECST_ELLIPSOID;
                            coll->Shapes.push_back(shape);
                        }
                        break;
                    case SCollisionShapeDef::ECST_TRIANGLE:
                        {
                            STriangleCollider* tmp = static_cast<STriangleCollider*>(Shapes[i].object);
                            SCollisionShapeDef shape;
                            shape.object = new STriangleCollider(*tmp);
                            shape.objectType = SCollisionShapeDef::ECST_TRIANGLE;
                            coll->Shapes.push_back(shape);
                        }
                        break;
                    case SCollisionShapeDef::ECST_TRIANGLE_MESH:
                        {
                            STriangleMeshCollider* tmp = static_cast<STriangleMeshCollider*>(Shapes[i].object);
                            coll->AddTriangleMesh(tmp);
                        }
                        break;
                    case SCollisionShapeDef::ECST_COUNT:
                        assert(0);
                        break;
                }
            }
            return coll;
        }

		//! Performs collision test with given ray.
		/**
		@param[out] collisionDistance Distance between collider and ray.
		@param[in] origin Attachment point of the ray.
		@param[in] direction Normalized drection vector of the ray.
		@param[in] dirMaxMultiplier
		*/
        inline bool CollideWithRay(float& collisionDistance, vectorSIMDf origin, vectorSIMDf direction, const float& dirMaxMultiplier) const
        {
            if (colliderData.attachedNode)
            {
                matrix4x3 absoluteTransform = colliderData.attachedNode->getAbsoluteTransformation();
                bool retval = absoluteTransform.makeInverse();
                if (!retval)
                    return false;

                absoluteTransform.transformVect(origin.pointer,origin.pointer);
                origin.pointer[3] = 0.f;
                absoluteTransform.mulSub3x3With3x1(direction.pointer,direction.pointer); /// Actually a 3x3 submatrix multiply

                switch (colliderData.attachedNode->getType())
                {
                    case scene::ESNT_MESH_INSTANCED:
                        {
                            core::matrix4x3 instanceTform = static_cast<scene::IMeshSceneNodeInstanced*>(colliderData.attachedNode)->getInstanceTransform(colliderData.instanceID);
                            retval = instanceTform.makeInverse();
                            if (!retval)
                                return false;

                            instanceTform.transformVect(origin.pointer,origin.pointer);
                            origin.pointer[3] = 0.f;
                            instanceTform.mulSub3x3With3x1(direction.pointer,direction.pointer);
                        }
                        break;
                    ///case ESNT_INSTANCED_ANIMATED_MESH:
                    default:
                        break;
                }
            }

            vectorSIMDf direction_reciprocal = reciprocal(direction);
            float dummyPosition;
            if (!BBox.CollideWithRay(dummyPosition,origin,direction,dirMaxMultiplier,direction_reciprocal))
            {
                return false;
            }


            for (size_t i=0; i<Shapes.size(); i++)
            {
                switch (Shapes[i].objectType)
                {
                    case SCollisionShapeDef::ECST_AABOX:
                        {
                            SAABoxCollider* tmp = static_cast<SAABoxCollider*>(Shapes[i].object);
                            if (tmp->CollideWithRay(collisionDistance,origin,direction,dirMaxMultiplier,direction_reciprocal))
                                return true;
                        }
                        break;
                    case SCollisionShapeDef::ECST_ELLIPSOID:
                        {
                            SEllipsoidCollider* tmp = static_cast<SEllipsoidCollider*>(Shapes[i].object);
                            if (tmp->CollideWithRay(collisionDistance,origin,direction,dirMaxMultiplier))
                                return true;
                        }
                        break;
                    case SCollisionShapeDef::ECST_TRIANGLE:
                        {
                            STriangleCollider* tmp = static_cast<STriangleCollider*>(Shapes[i].object);
                            if (tmp->CollideWithRay(collisionDistance,origin,direction,dirMaxMultiplier))
                                return true;
                        }
                        break;
                    case SCollisionShapeDef::ECST_TRIANGLE_MESH:
                        {
                            STriangleMeshCollider* tmp = static_cast<STriangleMeshCollider*>(Shapes[i].object);
                            if (tmp->CollideWithRay(collisionDistance,origin,direction,dirMaxMultiplier))
                                return true;
                        }
                        break;
                    case SCollisionShapeDef::ECST_COUNT:
                        assert(0);
                        break;
                }
            }
            return false;
        }

		inline size_t getShapeCount() const { return Shapes.size(); }
		inline const SAABoxCollider& getBoundingBox() const { return BBox; }
        inline const SColliderData& getColliderData() const {return colliderData;}

		//! Sets collider data.
		/** @param data The collider data.
		*/
        inline void setColliderData(const SColliderData& data)
        {
            colliderData = data;
        }

		//! Adds axis-aligned box collider.
		/** @param collider The box collider.
		@returns whether collider was succesfully added.
		*/
        inline bool AddBox(const SAABoxCollider& collider)
        {
            if (collider.Box.getVolume()<=0.f)
                return false;

            if (Shapes.size()==0)
                BBox = collider;
            else
                BBox.Box.addInternalBox(collider.Box);

            SAABoxCollider* tmp = new SAABoxCollider(collider);
            SCollisionShapeDef newShape;
            newShape.object = tmp;
            newShape.objectType = SCollisionShapeDef::ECST_AABOX;
            Shapes.push_back(newShape);
            return true;
        }
		//! Adds ellipsoid collider previosly creating it from center point and three axis lengths.
		/** @param center Center point (in 3d space) of the ellipsoid.
		@param axisLengths 3d vector denoting axis lengths of ellipsoid.
		@returns Whether collider was successfully added.
		*/
        inline bool AddEllipsoid(const vectorSIMDf& centr, const vectorSIMDf& axisLengths)
        {
            bool retval = true;
            SEllipsoidCollider* tmp = new SEllipsoidCollider(retval,centr,axisLengths);
            if (!retval)
            {
                delete tmp;
                return false;
            }

            if (Shapes.size()==0)
            {
                BBox.Box.MinEdge = (centr-axisLengths).getAsVector3df();
                BBox.Box.MaxEdge = (centr+axisLengths).getAsVector3df();
            }
            else
            {
                BBox.Box.addInternalPoint((centr-axisLengths).getAsVector3df());
                BBox.Box.addInternalPoint((centr+axisLengths).getAsVector3df());
            }

            SCollisionShapeDef newShape;
            newShape.object = tmp;
            newShape.objectType = SCollisionShapeDef::ECST_ELLIPSOID;
            Shapes.push_back(newShape);
            return true;
        }
		//! Adds triangle collider previously creating it from three given points.
		/**
		@param A First point.
		@param B Second point.
		@param C Third point.
		@returns Whether collider was successfully added.
		*/
        inline bool AddTriangle(const vectorSIMDf& A, const vectorSIMDf& B, const vectorSIMDf& C)
        {
            bool retval = true;
            STriangleCollider* tmp = new STriangleCollider(A,B,C,retval);
            if (!retval)
            {
                delete tmp;
                return false;
            }

            if (Shapes.size()==0)
                BBox.Box.reset(A.getAsVector3df());
            else
                BBox.Box.addInternalPoint(A.getAsVector3df());
            BBox.Box.addInternalPoint(B.getAsVector3df());
            BBox.Box.addInternalPoint(C.getAsVector3df());

            SCollisionShapeDef newShape;
            newShape.object = tmp;
            newShape.objectType = SCollisionShapeDef::ECST_TRIANGLE;
            Shapes.push_back(newShape);
            return true;
        }
		//! Adds triangle mesh collider.
		/** @param collider Pointer to triangle mesh collider.
		*/
        inline bool AddTriangleMesh(STriangleMeshCollider* collider)
        {
            if (collider->getTriangleCount()==0)
                return false;

            if (Shapes.size()==0)
                BBox = collider->getBoundingBox();
            else
                BBox.Box.addInternalBox(collider->getBoundingBox().Box);

            collider->grab();

            SCollisionShapeDef newShape;
            newShape.object = collider;
            newShape.objectType = SCollisionShapeDef::ECST_TRIANGLE_MESH;
            Shapes.push_back(newShape);
            return true;
        }
};


}
}

#endif
// documented by Krzysztof Szenk on 12-02-2018


