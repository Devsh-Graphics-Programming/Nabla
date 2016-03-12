// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_ANIMATED_MESH_H_INCLUDED__
#define __I_ANIMATED_MESH_H_INCLUDED__

#include "aabbox3d.h"
#include "IMesh.h"

namespace irr
{
namespace scene
{

	//! Interface for an animated mesh.
	/** There are already simple implementations of this interface available so
	you don't have to implement this interface on your own if you need to:
	You might want to use irr::scene::SAnimatedMesh, irr::scene::SMesh,
	irr::scene::SMeshBuffer etc. */
	template <class T>
	class IAnimatedMesh : public T
	{
	public:

		//! Gets the frame count of the animated mesh.
		/** \return The amount of frames. If the amount is 1,
		it is a static, non animated mesh. */
		virtual u32 getFrameCount() const = 0;

		//! Gets the animation speed of the animated mesh.
		/** \return The number of frames per second to play the
		animation with by default. If the amount is 0,
		it is a static, non animated mesh. */
		virtual f32 getAnimationSpeed() const = 0;

		//! Sets the animation speed of the animated mesh.
		/** \param fps Number of frames per second to play the
		animation with by default. If the amount is 0,
		it is not animated. The actual speed is set in the
		scene node the mesh is instantiated in.*/
		virtual void setAnimationSpeed(f32 fps) =0;

		//! Returns the IMesh interface for a frame.
		/** \param frame: Frame number as zero based index. The maximum
		frame number is getFrameCount() - 1;
		\param detailLevel: Level of detail. 0 is the lowest, 255 the
		highest level of detail. Most meshes will ignore the detail level.
		\return Returns the animated mesh based on a detail level. */
		virtual T* getMesh(s32 frame, s32 detailLevel=255) = 0;

		//! Returns the type of the animated mesh.
		/** In most cases it is not neccessary to use this method.
		This is useful for making a safe downcast.
		\returns Type of the mesh. */
		virtual E_MESH_TYPE getMeshType() const
		{
			return EMT_ANIMATED_FRAME_BASED;
		}
	};

	typedef IAnimatedMesh<IGPUMesh> IGPUAnimatedMesh;
	typedef IAnimatedMesh<ICPUMesh> ICPUAnimatedMesh;

} // end namespace scene
} // end namespace irr

#endif

