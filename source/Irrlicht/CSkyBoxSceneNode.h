// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_SKY_BOX_SCENE_NODE_H_INCLUDED__
#define __C_SKY_BOX_SCENE_NODE_H_INCLUDED__

#include "ISceneNode.h"

namespace irr
{
namespace scene
{

	// Skybox, rendered with zbuffer turned off, before all other nodes.
	class CSkyBoxSceneNode : public ISceneNode
	{
        protected:
            virtual ~CSkyBoxSceneNode()
            {
                for (size_t i=0; i<6; i++)
                    sides[i]->drop();
            }

        public:
            //! constructor
            CSkyBoxSceneNode(	core::smart_refctd_ptr<video::ITexture>&& top, core::smart_refctd_ptr<video::ITexture>&& bottom,
								core::smart_refctd_ptr<video::ITexture>&& left, core::smart_refctd_ptr<video::ITexture>&& right,
								core::smart_refctd_ptr<video::ITexture>&& front, core::smart_refctd_ptr<video::ITexture>&& back,
								core::smart_refctd_ptr<video::IGPUBuffer>&& vertPositions, size_t positionsOffsetInBuf,
								IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id);
            //! clone Ctor
            CSkyBoxSceneNode(CSkyBoxSceneNode* other,
                IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id);

            //!
            virtual bool supportsDriverFence() const {return true;}

            virtual void OnRegisterSceneNode();

            //! renders the node.
            virtual void render();

            //! returns the axis aligned bounding box of this node
            virtual const core::aabbox3d<float>& getBoundingBox();

            //! Returns type of the scene node
            virtual ESCENE_NODE_TYPE getType() const { return ESNT_SKY_BOX; }

            //! Creates a clone of this scene node and its children.
            virtual ISceneNode* clone(IDummyTransformationSceneNode* newParent=0, ISceneManager* newManager=0) { assert(false); return nullptr; }

        private:

            core::aabbox3d<float> Box;
            video::IGPUMeshBuffer* sides[6];
            video::SGPUMaterial Material[6];
	};

} // end namespace scene
} // end namespace irr

#endif

