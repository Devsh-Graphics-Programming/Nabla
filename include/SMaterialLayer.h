// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __S_MATERIAL_LAYER_H_INCLUDED__
#define __S_MATERIAL_LAYER_H_INCLUDED__

#include "STextureSamplingParams.h"
#include "irr/asset/ICPUTexture.h"

namespace irr
{
namespace video
{

#include "irr/irrpack.h"

	//! Struct for holding material parameters which exist per texture layer
    template<typename TexT>
	class SMaterialLayer
	{
	public:
		//! Default constructor
		SMaterialLayer()
			: Texture(0)
        {
        }

		//! Copy constructor
		/** \param other Material layer to copy from. */
		SMaterialLayer(const SMaterialLayer& other)
		{
			*this = other;
		}

		//! Destructor
		~SMaterialLayer()
		{
		}

		//! Assignment operator
		/** \param other Material layer to copy from.
		\return This material layer, updated. */
		SMaterialLayer& operator=(const SMaterialLayer& other)
		{
			// Check for self-assignment!
			if (this == &other)
				return *this;

			Texture = other.Texture;
			SamplingParams = other.SamplingParams;

			return *this;
		}

		//! Inequality operator
		/** \param b Layer to compare to.
		\return True if layers are different, else false. */
		inline bool operator!=(const SMaterialLayer& b) const
		{
			return Texture != b.Texture || SamplingParams.calculateHash(Texture) != b.SamplingParams.calculateHash(b.Texture);
		}

		//! Equality operator
		/** \param b Layer to compare to.
		\return True if layers are equal, else false. */
		inline bool operator==(const SMaterialLayer& b) const
		{ return !(b!=*this); }

		//! Texture
		TexT* Texture;

        STextureSamplingParams SamplingParams;
	} PACK_STRUCT;

#include "irr/irrunpack.h"

} // end namespace video
} // end namespace irr

#endif // __S_MATERIAL_LAYER_H_INCLUDED__
