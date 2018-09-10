// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_SCENE_NODE_ANIMATOR_TEXTURE_H_INCLUDED__
#define __C_SCENE_NODE_ANIMATOR_TEXTURE_H_INCLUDED__


#include "ISceneNodeAnimatorFinishing.h"

namespace irr
{
namespace video
{
    class ITexture;
}
namespace scene
{
	class CSceneNodeAnimatorTexture : public ISceneNodeAnimatorFinishing
	{
    protected:
		//! destructor
		virtual ~CSceneNodeAnimatorTexture();

	public:
		//! constructor
		CSceneNodeAnimatorTexture(const core::vector<video::ITexture*>& textures,
			int32_t timePerFrame, bool loop, uint32_t now);

		//! animates a scene node
		virtual void animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs);

		//! Returns type of the scene node animator
		virtual ESCENE_NODE_ANIMATOR_TYPE getType() const { return ESNAT_TEXTURE; }

		//! Creates a clone of this animator.
		/** Please note that you will have to drop
		(IReferenceCounted::drop()) the returned pointer after calling
		this. */
		virtual ISceneNodeAnimator* createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager=0);

	private:

		void clearTextures();

		core::vector<video::ITexture*> Textures;
		uint32_t TimePerFrame;
		uint32_t StartTime;
		bool Loop;
	};


} // end namespace scene
} // end namespace irr

#endif

