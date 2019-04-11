// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SKINNED_MESH_H_INCLUDED__
#define __I_SKINNED_MESH_H_INCLUDED__

#include "irr/core/Types.h"
#include "irr/video/IGPUMesh.h"
#include "quaternion.h"
#include "matrix4x3.h"
#include <vector>
#include <string>

namespace irr
{
namespace scene
{
    class CFinalBoneHierarchy;
}
namespace video
{

    class IGPUSkinnedMesh : public video::IGPUMesh
    {
        protected:
            virtual ~IGPUSkinnedMesh()
            {
                //referenceHierarchy drop in child classes
            }

            const scene::CFinalBoneHierarchy* referenceHierarchy;
            //! The bounding box of this mesh
            core::aabbox3d<float> Box;
        public:
            IGPUSkinnedMesh(scene::CFinalBoneHierarchy* boneHierarchy) : referenceHierarchy(boneHierarchy)
            {
                //referenceHierarchy grab in child classes
            }

            //!
            inline const scene::CFinalBoneHierarchy* getBoneReferenceHierarchy() const {return referenceHierarchy;}

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

            virtual asset::E_MESH_TYPE getMeshType() const
            {
                return asset::EMT_ANIMATED_SKINNED;
            }

            //! can use more efficient shaders this way :D
            virtual const uint32_t& getMaxVertexWeights(const size_t& meshbufferIx) const =0;

            //!
            virtual uint32_t getMaxVertexWeights() const =0;
    };

} // end namespace video
} // end namespace irr

#endif

