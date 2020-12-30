// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_E_SCENE_NODE_TYPES_H_INCLUDED__
#define __NBL_E_SCENE_NODE_TYPES_H_INCLUDED__

#include "nbl/core/Types.h"

namespace nbl
{
namespace scene
{

	//! An enumeration for all types of built-in scene nodes
	/** A scene node type is represented by a four character code
	such as 'cube' or 'mesh' instead of simple numbers, to avoid
	name clashes with external scene nodes.*/
	enum ESCENE_NODE_TYPE
	{
		//! of type CSceneManager (note that ISceneManager is not(!) an ISceneNode)
		ESNT_SCENE_MANAGER,

		//! Sky Box Scene Node
		ESNT_SKY_BOX,

		//! Sky Dome Scene Node
		ESNT_SKY_DOME,

		//! Mesh Scene Node
		ESNT_MESH,
		ESNT_MESH_INSTANCED,

		//! Dummy Transformation Scene Node
		ESNT_DUMMY_TRANSFORMATION,

		//! Camera Scene Node
		ESNT_CAMERA,

		//! Animated Mesh Scene Node
		ESNT_ANIMATED_MESH,
		ESNT_ANIMATED_MESH_INSTANCED,

		//! Skinned Mesh Scene Node
		ESNT_SKINNED_MESH,
		ESNT_SKINNED_MESH_INSTANCED,

		//! Maya Camera Scene Node
		/** Legacy, for loading version <= 1.4.x .irr files */
		ESNT_CAMERA_MAYA,

		//! First Person Shooter Camera
		/** Legacy, for loading version <= 1.4.x .irr files */
		ESNT_CAMERA_FPS,

		//! Unknown scene node
		ESNT_UNKNOWN,

		//! Will match with any scene node when checking types
		ESNT_ANY           
	};



} // end namespace scene
} // end namespace nbl


#endif

