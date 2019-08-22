// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SKINNED_MESH_H_INCLUDED__
#define __I_SKINNED_MESH_H_INCLUDED__

#include "irr/core/core.h"
#include "irr/video/IGPUMesh.h"

namespace irr
{
namespace video
{

    class IGPUSkinnedMesh : public video::IGPUMesh
    {
        protected:
			virtual ~IGPUSkinnedMesh() {}

            core::smart_refctd_ptr<const asset::CFinalBoneHierarchy> referenceHierarchy;
            //! The bounding box of this mesh
            core::aabbox3d<float> Box;
        public:
            IGPUSkinnedMesh(core::smart_refctd_ptr<const asset::CFinalBoneHierarchy>&& boneHierarchy) : referenceHierarchy(std::move(boneHierarchy))
            {
                //referenceHierarchy grab in child classes
            }

            //!
            inline const asset::CFinalBoneHierarchy* getBoneReferenceHierarchy() const {return referenceHierarchy.get();}

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
			//! Gets the frame count of the animated mesh.
			virtual uint32_t getFrameCount() const { return referenceHierarchy->getKeyFrameCount(); }
			virtual float getFirstFrame() const
			{
				if (referenceHierarchy->getKeyFrameCount())
					return referenceHierarchy->getKeys()[0];
				else
					return 0.f;
			}
			virtual float getLastFrame() const
			{
				if (referenceHierarchy->getKeyFrameCount())
					return referenceHierarchy->getKeys()[referenceHierarchy->getKeyFrameCount() - 1];
				else
					return 0.f;
			}

			virtual asset::E_MESH_TYPE getMeshType() const override
			{
				return asset::EMT_ANIMATED_SKINNED;
			}

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

