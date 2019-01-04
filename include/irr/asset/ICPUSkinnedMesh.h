#ifndef __IRR_I_CPU_SKINNED_MESH_INCLUDED__
#define __IRR_I_CPU_SKINNED_MESH_INCLUDED__

#include "ICPUMesh.h"
#include "quaternion.h"

namespace irr { namespace asset
{

//! Interface for using some special functions of Skinned meshes
class ICPUSkinnedMesh : public ICPUMesh
{
public:

    //!
    virtual scene::CFinalBoneHierarchy* getBoneReferenceHierarchy() const = 0;

    //!
    virtual asset::E_MESH_TYPE getMeshType() const
    {
        return asset::EMT_ANIMATED_SKINNED;
    }

    virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const
    {
        return asset::CorrespondingBlobTypeFor<ICPUSkinnedMesh>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
    }

    //! Animation keyframe which describes a new position
    class SPositionKey
    {
    public:
        SPositionKey() {}
        SPositionKey(const float& mockFrame) : frame(mockFrame) {}
        inline bool operator<(const SPositionKey& other) const { return (frame < other.frame); }

        float frame;
        core::vector3df position;
    };

    //! Animation keyframe which describes a new scale
    class SScaleKey
    {
    public:
        SScaleKey() {}
        SScaleKey(const float& mockFrame) : frame(mockFrame) {}
        inline bool operator<(const SScaleKey& other) const { return (frame < other.frame); }

        float frame;
        core::vector3df scale;
    };

    //! Animation keyframe which describes a new rotation
    class SRotationKey
    {
    public:
        SRotationKey() {}
        SRotationKey(const float& mockFrame) : frame(mockFrame) {}
        inline bool operator<(const SRotationKey& other) const { return (frame < other.frame); }

        float frame;
        core::quaternion rotation;
    };

    //! Joints
    class SJoint
    {
    public:
        SJoint() : Parent(NULL)
        {
        }

        //! The name of this joint
        std::string Name;

        //! Local matrix of this joint
        core::matrix4x3 LocalMatrix;

        //! List of child joints
        SJoint* Parent;
        core::vector<SJoint*> Children;


        inline SPositionKey* addPositionKey()
        {
            PositionKeys.push_back(SPositionKey());
            return &PositionKeys.back();
        }


        inline SScaleKey* addScaleKey()
        {
            ScaleKeys.push_back(SScaleKey());
            return &ScaleKeys.back();
        }


        inline SRotationKey* addRotationKey()
        {
            RotationKeys.push_back(SRotationKey());
            return &RotationKeys.back();
        }

        //! Animation keys causing translation change
        core::vector<SPositionKey> PositionKeys;

        //! Animation keys causing scale change
        core::vector<SScaleKey> ScaleKeys;

        //! Animation keys causing rotation change
        core::vector<SRotationKey> RotationKeys;

        //! Unnecessary for loaders, will be overwritten on finalize
        core::aabbox3df bbox;

        core::matrix4x3 GlobalMatrix;

        core::matrix4x3 GlobalInversedMatrix; //the x format pre-calculates this
    };


    //Interface for the mesh loaders (finalize should lock these functions, and they should have some prefix like loader_

    //these functions will use the needed arrays, set values, etc to help the loaders

    //! exposed for loaders: joints list
    virtual core::vector<SJoint*>& getAllJoints() = 0;

    //! exposed for loaders: joints list
    virtual const core::vector<SJoint*>& getAllJoints() const = 0;

    //! loaders should call this after populating the mesh
    virtual void finalize() = 0;

    //! Adds a new joint to the mesh, access it as last one
    virtual SJoint* addJoint(SJoint *parent = 0) = 0;

    //! Check if the mesh is non-animated
    virtual bool isStatic() const = 0;
};

}}//irr::asset

#endif //__IRR_I_CPU_SKINNED_MESH_INCLUDED__