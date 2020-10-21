// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSceneNodeAnimatorDelete.h"
#include "ISceneManager.h"

namespace irr
{
namespace scene
{


//! constructor
CSceneNodeAnimatorDelete::CSceneNodeAnimatorDelete(ISceneManager* manager, uint32_t time)
: ISceneNodeAnimatorFinishing(time), SceneManager(manager)
{
	#ifdef _NBL_DEBUG
	setDebugName("CSceneNodeAnimatorDelete");
	#endif
}


//! animates a scene node
void CSceneNodeAnimatorDelete::animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs)
{
	if (timeMs > FinishTime)
	{
		HasFinished = true;
		if(node && SceneManager)
            SceneManager->addToDeletionQueue(node);
	}
}


ISceneNodeAnimator* CSceneNodeAnimatorDelete::createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager)
{
	CSceneNodeAnimatorDelete * newAnimator =
		new CSceneNodeAnimatorDelete(SceneManager, FinishTime);

	return newAnimator;
}


} // end namespace scene
} // end namespace irr

