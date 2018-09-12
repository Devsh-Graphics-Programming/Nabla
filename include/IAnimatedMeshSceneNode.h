// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_ANIMATED_MESH_SCENE_NODE_H_INCLUDED__
#define __I_ANIMATED_MESH_SCENE_NODE_H_INCLUDED__

#include "ISceneNode.h"
#include "IAnimatedMesh.h"

namespace irr
{
namespace scene
{


	//! Callback interface for catching events of ended animations.
	/** Implement this interface to be able to
	be notified if an animation playback has ended.
	**/
	template<class T>
	class IAnimationEndCallBack : public virtual core::IReferenceCounted
	{
            _IRR_INTERFACE_CHILD(IAnimationEndCallBack) {}
        public:

            //! Will be called when the animation playback has ended.
            /** See ISkinnedMeshSceneNode::setAnimationEndCallback and
            IAnimatedMeshSceneNoe::setAnimationEndCallback for more informations.
            \param node: Node of which the animation has ended. */
            virtual void OnAnimationEnd(T* node) = 0;
	};

} // end namespace scene
} // end namespace irr

#endif

