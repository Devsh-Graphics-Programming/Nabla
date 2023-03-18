// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_COLOR_H_INCLUDED__
#define __NBL_COLOR_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "vectorSIMD.h"

#include "nbl/asset/format/decodePixels.h"

namespace nbl
{
namespace video
{

// nuke this
	//! Creates a 16 bit A1R5G5B5 color
	inline uint16_t RGBA16(uint32_t r, uint32_t g, uint32_t b, uint32_t a=0xFF)
	{
		return (uint16_t)((a & 0x80) << 8 |
			(r & 0xF8) << 7 |
			(g & 0xF8) << 2 |
			(b & 0xF8) >> 3);
	}


	//! Creates a 16 bit A1R5G5B5 color
	inline uint16_t RGB16(uint32_t r, uint32_t g, uint32_t b)
	{
		return RGBA16(r,g,b);
	}

	//! Converts a 32bit (X8R8G8B8) color to a 16bit A1R5G5B5 color
	inline uint16_t X8R8G8B8toA1R5G5B5(uint32_t color)
	{
		return (uint16_t)(0x8000 |
			( color & 0x00F80000) >> 9 |
			( color & 0x0000F800) >> 6 |
			( color & 0x000000F8) >> 3);
	}

	//! Convert A8R8G8B8 Color from A1R5G5B5 color
	/** build a nicer 32bit Color by extending dest lower bits with source high bits. */
	inline uint32_t A1R5G5B5toA8R8G8B8(uint16_t color)
	{
		return ( (( -( (int32_t) color & 0x00008000 ) >> (int32_t) 31 ) & 0xFF000000 ) |
				(( color & 0x00007C00 ) << 9) | (( color & 0x00007000 ) << 4) |
				(( color & 0x000003E0 ) << 6) | (( color & 0x00000380 ) << 1) |
				(( color & 0x0000001F ) << 3) | (( color & 0x0000001C ) >> 2)
				);
	}

	//! Class representing a 32 bit ARGB color.
	/** The color values for alpha, red, green, and blue are
	stored in a single uint32_t. So all four values may be between 0 and 255.
	Alpha in Irrlicht is opacity, so 0 is fully transparent, 255 is fully opaque (solid).
	This class is used by most parts of the Irrlicht Engine
	to specify a color. Another way is using the class SColorf, which
	stores the color values in 4 floats.
	This class must consist of only one uint32_t and must not use virtual functions.
	*/
	class SColor
	{
	public:

		//! Constructor of the Color. Does nothing.
		/** The color value is not initialized to save time. */
		SColor() {}

		//! Constructs the color from 4 values representing the alpha, red, green and blue component.
		/** Must be values between 0 and 255. */
		SColor (uint32_t a, uint32_t r, uint32_t g, uint32_t b)
			: color(((a & 0xff)<<24) | ((r & 0xff)<<16) | ((g & 0xff)<<8) | (b & 0xff)) {}

		//! Constructs the color from a 32 bit value. Could be another color.
		SColor(uint32_t clr)
			: color(clr) {}

		//! Returns the alpha component of the color.
		/** The alpha component defines how opaque a color is.
		\return The alpha value of the color. 0 is fully transparent, 255 is fully opaque. */
		uint32_t getAlpha() const { return color>>24; }

		//! Returns the red component of the color.
		/** \return Value between 0 and 255, specifying how red the color is.
		0 means no red, 255 means full red. */
		uint32_t getRed() const { return (color>>16) & 0xff; }

		//! Returns the green component of the color.
		/** \return Value between 0 and 255, specifying how green the color is.
		0 means no green, 255 means full green. */
		uint32_t getGreen() const { return (color>>8) & 0xff; }

		//! Returns the blue component of the color.
		/** \return Value between 0 and 255, specifying how blue the color is.
		0 means no blue, 255 means full blue. */
		uint32_t getBlue() const { return color & 0xff; }

		//! Sets the alpha component of the Color.
		/** The alpha component defines how transparent a color should be.
		\param a The alpha value of the color. 0 is fully transparent, 255 is fully opaque. */
		void setAlpha(uint32_t a) { color = ((a & 0xff)<<24) | (color & 0x00ffffff); }

		//! Sets the red component of the Color.
		/** \param r: Has to be a value between 0 and 255.
		0 means no red, 255 means full red. */
		void setRed(uint32_t r) { color = ((r & 0xff)<<16) | (color & 0xff00ffff); }

		//! Sets the green component of the Color.
		/** \param g: Has to be a value between 0 and 255.
		0 means no green, 255 means full green. */
		void setGreen(uint32_t g) { color = ((g & 0xff)<<8) | (color & 0xffff00ff); }

		//! Sets the blue component of the Color.
		/** \param b: Has to be a value between 0 and 255.
		0 means no blue, 255 means full blue. */
		void setBlue(uint32_t b) { color = (b & 0xff) | (color & 0xffffff00); }

		//! Converts color to OpenGL color format
		/** From ARGB to RGBA in 4 byte components for endian aware
		passing to OpenGL
		\param dest: address where the 4x8 bit OpenGL color is stored. */
		void toOpenGLColor(uint8_t* dest) const
		{
			*dest =   (uint8_t)getRed();
			*++dest = (uint8_t)getGreen();
			*++dest = (uint8_t)getBlue();
			*++dest = (uint8_t)getAlpha();
		}

		//! Sets all four components of the color at once.
		/** Constructs the color from 4 values representing the alpha,
		red, green and blue components of the color. Must be values
		between 0 and 255.
		\param a: Alpha component of the color. The alpha component
		defines how transparent a color should be. Has to be a value
		between 0 and 255. 255 means not transparent (opaque), 0 means
		fully transparent.
		\param r: Sets the red component of the Color. Has to be a
		value between 0 and 255. 0 means no red, 255 means full red.
		\param g: Sets the green component of the Color. Has to be a
		value between 0 and 255. 0 means no green, 255 means full
		green.
		\param b: Sets the blue component of the Color. Has to be a
		value between 0 and 255. 0 means no blue, 255 means full blue. */
		void set(uint32_t a, uint32_t r, uint32_t g, uint32_t b)
		{
			color = (((a & 0xff)<<24) | ((r & 0xff)<<16) | ((g & 0xff)<<8) | (b & 0xff));
		}
		void set(uint32_t col) { color = col; }

		//! Compares the color to another color.
		/** \return True if the colors are the same, and false if not. */
		bool operator==(const SColor& other) const { return other.color == color; }

		//! Compares the color to another color.
		/** \return True if the colors are different, and false if they are the same. */
		bool operator!=(const SColor& other) const { return other.color != color; }

		//! comparison operator
		/** \return True if this color is smaller than the other one */
		bool operator<(const SColor& other) const { return (color < other.color); }

		//! color in A8R8G8B8 Format
		uint32_t color;
	};

// nuke end

	//! Class representing a color with four floats.
	/** The color values for red, green, blue
	and alpha are each stored in a 32 bit floating point variable.
	So all four values may be between 0.0f and 1.0f.
	Another, faster way to define colors is using the class SColor, which
	stores the color values in a single 32 bit integer.
	*/
	class SColorf : private core::vectorSIMDf
	{
	public:
		//! Default constructor for SColorf.
		/** Sets red, green and blue to 0.0f and alpha to 1.0f. */
		SColorf() : vectorSIMDf(0.f,0.f,0.f,1.f) {}

		//! Constructs a color from up to four color values: red, green, blue, and alpha.
		/** \param r: Red color component. Should be a value between
		0.0f meaning no red and 1.0f, meaning full red.
		\param g: Green color component. Should be a value between 0.0f
		meaning no green and 1.0f, meaning full green.
		\param b: Blue color component. Should be a value between 0.0f
		meaning no blue and 1.0f, meaning full blue.
		\param a: Alpha color component of the color. The alpha
		component defines how transparent a color should be. Has to be
		a value between 0.0f and 1.0f, 1.0f means not transparent
		(opaque), 0.0f means fully transparent. */
		SColorf(float r_in, float g_in, float b_in, float a_in = 1.0f) : vectorSIMDf(r_in,g_in,b_in,a_in) {}

		SColorf(const vectorSIMDf& fromVec4) : vectorSIMDf(fromVec4) {}

		//! Constructs a color from 32 bit Color.
		/** \param c: 32 bit color from which this SColorf class is
		constructed from. */
		SColorf(SColor c)
		{
			r = static_cast<float>(c.getRed());
			g = static_cast<float>(c.getGreen());
			b = static_cast<float>(c.getBlue());
			a = static_cast<float>(c.getAlpha());

			const float inv = 1.0f / 255.0f;
			*this *= inv;
		}
		
		inline static SColorf fromSRGB(SColorf&& input)
		{
			float color[3] = {input.r, input.g, input.b};
			asset::impl::SRGB2lin<float>(color);
			
			return SColorf(color[0], color[1], color[2], input.getAlpha());
		}
		
		inline static SColorf toSRGB(SColorf&& input)
		{
			float color[3] = {input.r, input.g, input.b};
			asset::impl::lin2SRGB<float>(color);
			
			return SColorf(color[0], color[1], color[2], input.getAlpha());
		}

		//! Converts this color to a SColor without floats.
		inline SColor toSColor() const
		{
		    vectorSIMDf tmp = (*this) * 255.f;

			return SColor(core::round<float, uint32_t>(tmp.a), core::round<float, uint32_t>(tmp.r), core::round<float, uint32_t>(tmp.g), core::round<float, uint32_t>(tmp.b));
		}

		//! Sets three color components to new values at once.
		/** \param rr: Red color component. Should be a value between 0.0f meaning
		no red (=black) and 1.0f, meaning full red.
		\param gg: Green color component. Should be a value between 0.0f meaning
		no green (=black) and 1.0f, meaning full green.
		\param bb: Blue color component. Should be a value between 0.0f meaning
		no blue (=black) and 1.0f, meaning full blue. */
		inline void set(float rr, float gg, float bb) {r = rr; g =gg; b = bb; }

		//! Sets all four color components to new values at once.
		/** \param aa: Alpha component. Should be a value between 0.0f meaning
		fully transparent and 1.0f, meaning opaque.
		\param rr: Red color component. Should be a value between 0.0f meaning
		no red and 1.0f, meaning full red.
		\param gg: Green color component. Should be a value between 0.0f meaning
		no green and 1.0f, meaning full green.
		\param bb: Blue color component. Should be a value between 0.0f meaning
		no blue and 1.0f, meaning full blue. */
		inline void set(float aa, float rr, float gg, float bb) {a = aa; r = rr; g =gg; b = bb; }

		//! Returns the alpha component of the color in the range 0.0 (transparent) to 1.0 (opaque)
		inline float getAlpha() const { return a; }

		//! Returns the red component of the color in the range 0.0 to 1.0
		inline float getRed() const { return r; }

		//! Returns the green component of the color in the range 0.0 to 1.0
		inline float getGreen() const { return g; }

		//! Returns the blue component of the color in the range 0.0 to 1.0
		inline float getBlue() const { return b; }


		//!
		inline vectorSIMDf& getAsVectorSIMDf() {return *this;}
		inline const vectorSIMDf& getAsVectorSIMDf() const {return *this;}
	};


} // end namespace video
} // end namespace nbl

#endif