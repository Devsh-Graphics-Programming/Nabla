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
CSceneNodeAnimatorTexture::CSceneNodeAnimatorTexture(const core::array<video::ITexture*>& textures,
					 s32 timePerFrame, bool loop, u32 now)
: ISceneNodeAnimatorFinishing(0),
	TimePerFrame(timePerFrame), StartTime(now), Loop(loop)
{
	#ifdef _DEBUG
	setDebugName("CSceneNodeAnimatorTexture");
	#endif

	for (u32 i=0; i<textures.size(); ++i)
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
	for (u32 i=0; i<Textures.size(); ++i)
		if (Textures[i])
			Textures[i]->drop();
}


//! animates a scene node
void CSceneNodeAnimatorTexture::animateNode(IDummyTransformationSceneNode* node, u32 timeMs)
{
	if(!node)
		return;

	if (Textures.size())
	{
		const u32 t = (timeMs-StartTime);

		u32 idx = 0;
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

