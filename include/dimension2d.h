// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_DIMENSION2D_H_INCLUDED__
#define __NBL_DIMENSION2D_H_INCLUDED__

#include "BuildConfigOptions.h"
#include <cstdint>

namespace nbl
{
namespace core
{
template<class T>
class vector2d;

//! Specifies a 2 dimensional size.
template<class T>
class dimension2d  // : public AllocationOverrideDefault
{
public:
    //! Default constructor for empty dimension
    dimension2d()
        : Width(0), Height(0) {}
    //! Constructor with width and height
    dimension2d(const T& width, const T& height)
        : Width(width), Height(height) {}

    dimension2d(const vector2d<T>& other);  // Defined in vector2d.h

    //! Use this constructor only where you are sure that the conversion is valid.
    template<class U>
    explicit dimension2d(const dimension2d<U>& other)
        : Width((T)other.Width), Height((T)other.Height) {}

    template<class U>
    dimension2d<T>& operator=(const dimension2d<U>& other)
    {
        Width = (T)other.Width;
        Height = (T)other.Height;
        return *this;
    }

    //! Equality operator
    bool operator==(const dimension2d<T>& other) const
    {
        return Width == other.Width && Height == other.Height;
    }

    //! Inequality operator
    bool operator!=(const dimension2d<T>& other) const
    {
        return !(*this == other);
    }

    bool operator==(const vector2d<T>& other) const;  // Defined in vector2d.h

    bool operator!=(const vector2d<T>& other) const
    {
        return !(*this == other);
    }

    //! Set to new values
    dimension2d<T>& set(const T& width, const T& height)
    {
        Width = width;
        Height = height;
        return *this;
    }

    //! Divide width and height by scalar
    dimension2d<T>& operator/=(const T& scale)
    {
        Width /= scale;
        Height /= scale;
        return *this;
    }

    //! Divide width and height by scalar
    dimension2d<T> operator/(const T& scale) const
    {
        return dimension2d<T>(Width / scale, Height / scale);
    }

    //! Multiply width and height by scalar
    dimension2d<T>& operator*=(const T& scale)
    {
        Width *= scale;
        Height *= scale;
        return *this;
    }

    //! Multiply width and height by scalar
    dimension2d<T> operator*(const T& scale) const
    {
        return dimension2d<T>(Width * scale, Height * scale);
    }

    //! Add another dimension to this one.
    dimension2d<T>& operator+=(const dimension2d<T>& other)
    {
        Width += other.Width;
        Height += other.Height;
        return *this;
    }

    //! Add two dimensions
    dimension2d<T> operator+(const dimension2d<T>& other) const
    {
        return dimension2d<T>(Width + other.Width, Height + other.Height);
    }

    //! Subtract a dimension from this one
    dimension2d<T>& operator-=(const dimension2d<T>& other)
    {
        Width -= other.Width;
        Height -= other.Height;
        return *this;
    }

    //! Subtract one dimension from another
    dimension2d<T> operator-(const dimension2d<T>& other) const
    {
        return dimension2d<T>(Width - other.Width, Height - other.Height);
    }

    //! Get area
    T getArea() const
    {
        return Width * Height;
    }

    //! Get the optimal size according to some properties
    /** This is a function often used for texture dimension
			calculations. The function returns the next larger or
			smaller dimension which is a power-of-two dimension
			(2^n,2^m) and/or square (Width=Height).
			\param requirePowerOfTwo Forces the result to use only
			powers of two as values.
			\param requireSquare Makes width==height in the result
			\param larger Choose whether the result is larger or
			smaller than the current dimension. If one dimension
			need not be changed it is kept with any value of larger.
			\param maxValue Maximum texturesize. if value > 0 size is
			clamped to maxValue
			\return The optimal dimension under the given
			constraints. */
    dimension2d<T> getOptimalSize(
        bool requirePowerOfTwo = true,
        bool requireSquare = false,
        bool larger = true,
        uint32_t maxValue = 0) const
    {
        uint32_t i = 1;
        uint32_t j = 1;
        if(requirePowerOfTwo)
        {
            while(i < (uint32_t)Width)
                i <<= 1;
            if(!larger && i != 1 && i != (uint32_t)Width)
                i >>= 1;
            while(j < (uint32_t)Height)
                j <<= 1;
            if(!larger && j != 1 && j != (uint32_t)Height)
                j >>= 1;
        }
        else
        {
            i = (uint32_t)Width;
            j = (uint32_t)Height;
        }

        if(requireSquare)
        {
            if((larger && (i > j)) || (!larger && (i < j)))
                j = i;
            else
                i = j;
        }

        if(maxValue > 0 && i > maxValue)
            i = maxValue;

        if(maxValue > 0 && j > maxValue)
            j = maxValue;

        return dimension2d<T>((T)i, (T)j);
    }

    //! Width of the dimension.
    T Width;
    //! Height of the dimension.
    T Height;
};

//! Typedef for an float dimension.
typedef dimension2d<float> dimension2df;
//! Typedef for an unsigned integer dimension.
typedef dimension2d<uint32_t> dimension2du;

//! Typedef for an integer dimension.
/** There are few cases where negative dimensions make sense. Please consider using
		dimension2du instead. */
typedef dimension2d<int32_t> dimension2di;

}  // end namespace core
}  // end namespace nbl

#endif
