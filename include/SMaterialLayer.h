// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __S_MATERIAL_LAYER_H_INCLUDED__
#define __S_MATERIAL_LAYER_H_INCLUDED__

#include "irr/asset/ICPUTexture.h"
#include "STextureSamplingParams.h"
#include "IRenderableVirtualTexture.h"

namespace irr
{
namespace video
{

#include "irr/irrpack.h"

//! Struct for holding material parameters which exist per texture layer
template<typename TexT>
struct SMaterialLayer
{
	//! Default constructor
	SMaterialLayer() : Texture()
    {
    }

	//! Copy constructor
	/** \param other Material layer to copy from. */
	SMaterialLayer(const SMaterialLayer& other) : SMaterialLayer()
	{
		operator=(other);
	}
	//! Move constructor
	/** \param other Material layer to move. */
	SMaterialLayer(SMaterialLayer&& other) : SMaterialLayer()
	{
		operator=(std::move(other));
	}

	//! Assignment operator
	/** \param other Material layer to copy from.
	\return This material layer, updated. */
	SMaterialLayer& operator=(SMaterialLayer&& other)
	{
		std::swap(Texture,other.Texture);
		std::swap(SamplingParams,other.SamplingParams);

		return *this;
	}
	SMaterialLayer& operator=(const SMaterialLayer& other)
	{
		Texture = other.Texture;
		SamplingParams = other.SamplingParams;

		return *this;
	}

	//! Inequality operator
	/** \param b Layer to compare to.
	\return True if layers are different, else false. */
	inline bool operator!=(const SMaterialLayer& b) const
	{
		return Texture != b.Texture || SamplingParams.calculateHash(Texture.get()) != b.SamplingParams.calculateHash(b.Texture.get());
	}

	//! Equality operator
	/** \param b Layer to compare to.
	\return True if layers are equal, else false. */
	inline bool operator==(const SMaterialLayer& b) const
	{ return !(b!=*this); }

	//! Texture
	core::smart_refctd_ptr<TexT> Texture;

    STextureSamplingParams SamplingParams;
} PACK_STRUCT;

#include "irr/irrunpack.h"

} // end namespace video
} // end namespace irr

#endif // __S_MATERIAL_LAYER_H_INCLUDED__
