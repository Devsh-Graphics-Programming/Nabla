// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SKINNED_MESH_H_INCLUDED__
#define __I_SKINNED_MESH_H_INCLUDED__

#include "irr/core/Types.h"
#include "IMesh.h"
#include "quaternion.h"
#include "matrix4.h"
#include <vector>
#include <string>

namespace irr
{
namespace scene
{
    class CFinalBoneHierarchy;

    class IGPUSkinnedMesh : public IGPUMesh
    {
        protected:
            virtual ~IGPUSkinnedMesh()
            {
                //referenceHierarchy drop in child classes
            }

            const CFinalBoneHierarchy* referenceHierarchy;
            //! The bounding box of this mesh
            core::aabbox3d<float> Box;
        public:
            IGPUSkinnedMesh(CFinalBoneHierarchy* boneHierarchy) : referenceHierarchy(boneHierarchy)
            {
                //referenceHierarchy grab in child classes
            }

            //!
            inline const CFinalBoneHierarchy* getBoneReferenceHierarchy() const {return referenceHierarchy;}

            //! Returns an axis aligned bounding box of the mesh.
            /** \return A bounding box of this mesh is returned. */
            virtual const core::aabbox3d<float>& getBoundingBox() const
            {
                return Box;
            }

            //! set user axis aligned bounding box
            virtual void setBoundingBox(const core::aabbox3df& box)
            {
                Box = box;
            }

            //! Gets the frame count of the animated mesh.
            /** \return The amount of frames. If the amount is 1,
            it is a static, non animated mesh.
            If 0 it just is in the bind-pose doesn't have keyframes */
            virtual uint32_t getFrameCount() const =0;
            virtual float getFirstFrame() const =0;
            virtual float getLastFrame() const =0;

            virtual E_MESH_TYPE getMeshType() const
            {
                return EMT_ANIMATED_SKINNED;
            }

            //! can use more efficient shaders this way :D
            virtual const uint32_t& getMaxVertexWeights(const size_t& meshbufferIx) const =0;

            //!
            virtual uint32_t getMaxVertexWeights() const =0;
    };

	//! Interface for using some special functions of Skinned meshes
	class ICPUSkinnedMesh : public ICPUMesh
	{
	public:

		//!
		virtual CFinalBoneHierarchy* getBoneReferenceHierarchy() const = 0;

		//!
		virtual E_MESH_TYPE getMeshType() const
		{
			return EMT_ANIMATED_SKINNED;
		}

		virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const
		{
			return core::CorrespondingBlobTypeFor<ICPUSkinnedMesh>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
		}

		//! Animation keyframe which describes a new position
		struct SPositionKey
		{
            SPositionKey() {}
            SPositionKey(const float& mockFrame) : frame(mockFrame) {}
            inline bool operator<(const SPositionKey& other) const { return (frame < other.frame); }

            float frame;
            core::vector3df position;
		};

		//! Animation keyframe which describes a new scale
		struct SScaleKey
        {
            SScaleKey() {}
            SScaleKey(const float& mockFrame) : frame(mockFrame) {}
            inline bool operator<(const SScaleKey& other) const { return (frame < other.frame); }

			float frame;
			core::vector3df scale;
		};

		//! Animation keyframe which describes a new rotation
		struct SRotationKey
		{
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
		virtual SJoint* addJoint(SJoint *parent=0) = 0;

		//! Check if the mesh is non-animated
		virtual bool isStatic() const=0;
	};

} // end namespace scene
} // end namespace irr

#endif

