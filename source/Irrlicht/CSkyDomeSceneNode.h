// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// Code for this scene node has been contributed by Anders la Cour-Harbo (alc)

#ifndef __C_SKY_DOME_SCENE_NODE_H_INCLUDED__
#define __C_SKY_DOME_SCENE_NODE_H_INCLUDED__

#include "ISceneNode.h"
#include "irr/video/IGPUMeshBuffer.h"

namespace irr
{
namespace scene
{


class CSkyDomeSceneNode : public ISceneNode
{
    protected:
		virtual ~CSkyDomeSceneNode();

	public:
		CSkyDomeSceneNode(video::IVirtualTexture* texture, uint32_t horiRes, uint32_t vertRes,
			float texturePercentage, float spherePercentage, float radius,
			IDummyTransformationSceneNode* parent, ISceneManager* smgr, int32_t id);
		CSkyDomeSceneNode(CSkyDomeSceneNode* other,
			IDummyTransformationSceneNode* parent, ISceneManager* smgr, int32_t id);

		//!
		virtual bool supportsDriverFence() const {return true;}

		virtual void OnRegisterSceneNode();
		virtual void render();
		virtual const core::aabbox3d<float>& getBoundingBox();
		virtual video::SGPUMaterial& getMaterial(uint32_t i);
		virtual uint32_t getMaterialCount() const;
		virtual ESCENE_NODE_TYPE getType() const { return ESNT_SKY_DOME; }

		virtual ISceneNode* clone(IDummyTransformationSceneNode* newParent=0, ISceneManager* newManager=0);

	private:

		void generateMesh();

        video::IGPUMeshBuffer* Buffer;

        core::aabbox3df BoundingBox;
		uint32_t HorizontalResolution, VerticalResolution;
		float TexturePercentage, SpherePercentage, Radius;
};


}
}

#endif

