// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __E_DRIVER_TYPES_H_INCLUDED__
#define __E_DRIVER_TYPES_H_INCLUDED__

namespace irr
{
namespace video
{

	//! An enum for all types of drivers the Irrlicht Engine supports.
	enum E_DRIVER_TYPE
	{
		//! Null driver, useful for applications to run the engine without visualisation.
		/** The null device is able to load textures, but does not
		render and display any graphics. */
		EDT_NULL,

		//! The Burning's Software Renderer, stripped down
		/** Basically it's the Burning's Software Renderer from vanilla Irrlicht.
		It was stripped down to only provide rendering simple textured polygons
		for a GUI rendering fallback when OpenGL is not available.
        It still can do bilinear texel filtering. This renderer has been written
		entirely by Thomas Alten, and backported by DevSH. */
		EDT_BURNINGSVIDEO,

		//! OpenGL device, available on most platforms.
		/** Performs hardware accelerated rendering of 3D and 2D
		primitives. */
		EDT_OPENGL,

		//! No driver, just for counting the elements
		EDT_COUNT
	};

} // end namespace video
} // end namespace irr


#endif

