// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __I_SCENE_NODE_ANIMATOR_FINISHING_H_INCLUDED__
#define __I_SCENE_NODE_ANIMATOR_FINISHING_H_INCLUDED__

#include "ISceneNode.h"

namespace irr
{
namespace scene
{
	//! This is an abstract base class for animators that have a discrete end time.
	class ISceneNodeAnimatorFinishing : public ISceneNodeAnimator
	{
	public:

		//! constructor
		ISceneNodeAnimatorFinishing(uint32_t finishTime)
			: FinishTime(finishTime), HasFinished(false) { }

		virtual bool hasFinished(void) const { return HasFinished; }

	protected:

		uint32_t FinishTime;
		bool HasFinished;
	};


} // end namespace scene
} // end namespace irr

#endif

