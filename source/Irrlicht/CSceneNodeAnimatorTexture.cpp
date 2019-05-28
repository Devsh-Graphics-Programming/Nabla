// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CSceneNodeAnimatorTexture.h"
#include "ITexture.h"

namespace irr
{
namespace scene
{


//! constructor
CSceneNodeAnimatorTexture::CSceneNodeAnimatorTexture(const core::vector<video::ITexture*>& textures,
					 int32_t timePerFrame, bool loop, uint32_t now)
: ISceneNodeAnimatorFinishing(0),
	TimePerFrame(timePerFrame), StartTime(now), Loop(loop)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSceneNodeAnimatorTexture");
	#endif

	for (uint32_t i=0; i<textures.size(); ++i)
	{
		if (textures[i])
			textures[i]->grab();

		Textures.push_back(textures[i]);
	}

	FinishTime = now + (timePerFrame * Textures.size());
}


//! destructor
CSceneNodeAnimatorTexture::~CSceneNodeAnimatorTexture()
{
	clearTextures();
}


void CSceneNodeAnimatorTexture::clearTextures()
{
	for (uint32_t i=0; i<Textures.size(); ++i)
		if (Textures[i])
			Textures[i]->drop();
}


//! animates a scene node
void CSceneNodeAnimatorTexture::animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs)
{
	if(!node)
		return;

	if (Textures.size())
	{
		const uint32_t t = (timeMs-StartTime);

		uint32_t idx = 0;
		if (!Loop && timeMs >= FinishTime)
		{
			idx = Textures.size() - 1;
			HasFinished = true;
		}
		else
		{
			idx = (t/TimePerFrame) % Textures.size();
		}

		if (idx < Textures.size())
        {
            if (node->isISceneNode())
                static_cast<ISceneNode*>(node)->setMaterialTexture(0, Textures[idx]);
        }
	}
}



ISceneNodeAnimator* CSceneNodeAnimatorTexture::createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager)
{
	CSceneNodeAnimatorTexture * newAnimator =
		new CSceneNodeAnimatorTexture(Textures, TimePerFrame, Loop, StartTime);

	return newAnimator;
}


} // end namespace scene
} // end namespace irr

