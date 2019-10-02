// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_POINT_2D_H_INCLUDED__
#define __IRR_POINT_2D_H_INCLUDED__

#include "dimension2d.h"

namespace irr
{
namespace core
{


//! 2d vector template class with lots of operators and methods.
/** As of Irrlicht 1.6, this class supercedes position2d, which should
	be considered deprecated. */
template <class T>
class vector2d// : public AllocationOverrideDefault
{
public:
	//! Default constructor (null vector)
	vector2d() : X(0), Y(0) {}
	//! Constructor with two different values
	vector2d(T nx, T ny) : X(nx), Y(ny) {}
	//! Constructor with the same value for both members
	explicit vector2d(T n) : X(n), Y(n) {}
	//! Copy constructor
	vector2d(const vector2d<T>& other) : X(other.X), Y(other.Y) {}

	vector2d(const dimension2d<T>& other) : X(other.Width), Y(other.Height) {}

	// operators

	vector2d<T> operator-() const { return vector2d<T>(-X, -Y); }

	vector2d<T>& operator=(const vector2d<T>& other) { X = other.X; Y = other.Y; return *this; }

	vector2d<T>& operator=(const dimension2d<T>& other) { X = other.Width; Y = other.Height; return *this; }

	vector2d<T> operator+(const vector2d<T>& other) const { return vector2d<T>(X + other.X, Y + other.Y); }
	vector2d<T> operator+(const dimension2d<T>& other) const { return vector2d<T>(X + other.Width, Y + other.Height); }
	vector2d<T>& operator+=(const vector2d<T>& other) { X+=other.X; Y+=other.Y; return *this; }
	vector2d<T> operator+(const T v) const { return vector2d<T>(X + v, Y + v); }
	vector2d<T>& operator+=(const T v) { X+=v; Y+=v; return *this; }
	vector2d<T>& operator+=(const dimension2d<T>& other) { X += other.Width; Y += other.Height; return *this;  }

	vector2d<T> operator-(const vector2d<T>& other) const { return vector2d<T>(X - other.X, Y - other.Y); }
	vector2d<T> operator-(const dimension2d<T>& other) const { return vector2d<T>(X - other.Width, Y - other.Height); }
	vector2d<T>& operator-=(const vector2d<T>& other) { X-=other.X; Y-=other.Y; return *this; }
	vector2d<T> operator-(const T v) const { return vector2d<T>(X - v, Y - v); }
	vector2d<T>& operator-=(const T v) { X-=v; Y-=v; return *this; }
	vector2d<T>& operator-=(const dimension2d<T>& other) { X -= other.Width; Y -= other.Height; return *this;  }

	vector2d<T> operator*(const vector2d<T>& other) const { return vector2d<T>(X * other.X, Y * other.Y); }
	vector2d<T>& operator*=(const vector2d<T>& other) { X*=other.X; Y*=other.Y; return *this; }
	vector2d<T> operator*(const T v) const { return vector2d<T>(X * v, Y * v); }
	vector2d<T>& operator*=(const T v) { X*=v; Y*=v; return *this; }

	vector2d<T> operator/(const vector2d<T>& other) const { return vector2d<T>(X / other.X, Y / other.Y); }
	vector2d<T>& operator/=(const vector2d<T>& other) { X/=other.X; Y/=other.Y; return *this; }
	vector2d<T> operator/(const T v) const { return vector2d<T>(X / v, Y / v); }
	vector2d<T>& operator/=(const T v) { X/=v; Y/=v; return *this; }

	//! sort in order X, Y. Equality with rounding tolerance.
	bool operator<=(const vector2d<T>&other) const
	{
		return 	X<=other.X ||
				(X==other.X && Y<=other.Y);
	}

	//! sort in order X, Y. Equality with rounding tolerance.
	bool operator>=(const vector2d<T>&other) const
	{
		return 	X>=other.X || 
				(X==other.X && Y>=other.Y);
	}

	//! sort in order X, Y. Difference must be above rounding tolerance.
	bool operator<(const vector2d<T>&other) const
	{
		return 	X<other.X ||
				X==other.X && Y<other.Y;
	}

	//! sort in order X, Y. Difference must be above rounding tolerance.
	bool operator>(const vector2d<T>&other) const
	{
		return 	X>other.X ||
				X==other.X && Y>other.Y;
	}

	bool operator==(const vector2d<T>& other) const { return !operator!=(other); }
	bool operator!=(const vector2d<T>& other) const { return X!=other.X||Y!=other.Y; }

	// functions

	vector2d<T>& set(T nx, T ny) {X=nx; Y=ny; return *this; }
	vector2d<T>& set(const vector2d<T>& p) { X=p.X; Y=p.Y; return *this; }
/*
	T getLength() const { return core::squareroot( X*X + Y*Y ); }
	T getLengthSQ() const { return X*X + Y*Y; }
	T dotProduct(const vector2d<T>& other) const
	{
		return X*other.X + Y*other.Y;
	}
	T getDistanceFrom(const vector2d<T>& other) const
	{
		return vector2d<T>(X - other.X, Y - other.Y).getLength();
	}
	T getDistanceFromSQ(const vector2d<T>& other) const
	{
		return vector2d<T>(X - other.X, Y - other.Y).getLengthSQ();
	}
*/
	//! rotates the point anticlockwise around a center by an amount of degrees.
	/** \param degrees Amount of degrees to rotate by, anticlockwise.
	\param center Rotation center.
	\return This vector after transformation. */
	vector2d<T>& rotateBy(double degrees, const vector2d<T>& center=vector2d<T>())
	{
		degrees *= DEGTORAD64;
		const double cs = cos(degrees);
		const double sn = sin(degrees);

		X -= center.X;
		Y -= center.Y;

		set((T)(X*cs - Y*sn), (T)(X*sn + Y*cs));

		X += center.X;
		Y += center.Y;
		return *this;
	}

	//! Calculates the angle of this vector in degrees in the trigonometric sense.
	/** 0 is to the right (3 o'clock), values increase counter-clockwise.
	This method has been suggested by Pr3t3nd3r.
	\return Returns a value between 0 and 360. */
	double getAngleTrig() const
	{
		if (Y == 0)
			return X < 0 ? 180 : 0;
		else
		if (X == 0)
			return Y < 0 ? 270 : 90;

		if ( Y > 0)
			if (X > 0)
				return atan((double)Y/(double)X) * RADTODEG64;
			else
				return 180.0-atan((double)Y/-(double)X) * RADTODEG64;
		else
			if (X > 0)
				return 360.0-atan(-(double)Y/(double)X) * RADTODEG64;
			else
				return 180.0+atan(-(double)Y/-(double)X) * RADTODEG64;
	}

	//! Calculates the angle of this vector in degrees in the counter trigonometric sense.
	/** 0 is to the right (3 o'clock), values increase clockwise.
	\return Returns a value between 0 and 360. */
	inline double getAngle() const
	{
		if (Y == 0) // corrected thanks to a suggestion by Jox
			return X < 0 ? 180 : 0;
		else if (X == 0)
			return Y < 0 ? 90 : 270;

		// don't use getLength here to avoid precision loss with int32_t vectors
		// avoid floating-point trouble as sqrt(y*y) is occasionally larger than y, so clamp
		const double tmp = core::clamp(Y / std::sqrt((double)(X*X + Y*Y)), -1.0, 1.0);
		const double angle = atan( core::squareroot(1 - tmp*tmp) / tmp) * RADTODEG64;

		if (X>0 && Y>0)
			return angle + 270;
		else
		if (X>0 && Y<0)
			return angle + 90;
		else
		if (X<0 && Y<0)
			return 90 - angle;
		else
		if (X<0 && Y>0)
			return 270 - angle;

		return angle;
	}

	//! Calculates the angle between this vector and another one in degree.
	/** \param b Other vector to test with.
	\return Returns a value between 0 and 90. */
	inline double getAngleWith(const vector2d<T>& b) const
	{
		double tmp = (double)(X*b.X + Y*b.Y);

		if (tmp == 0.0)
			return 90.0;

		tmp = tmp / core::squareroot((double)((X*X + Y*Y) * (b.X*b.X + b.Y*b.Y)));
		if (tmp < 0.0)
			tmp = -tmp;
		if ( tmp > 1.0 ) //   avoid floating-point trouble
			tmp = 1.0;

		return atan(std::sqrt(1 - tmp*tmp) / tmp) * RADTODEG64;
	}

	//! Returns if this vector interpreted as a point is on a line between two other points.
	/** It is assumed that the point is on the line.
	\param begin Beginning vector to compare between.
	\param end Ending vector to compare between.
	\return True if this vector is between begin and end, false if not. */
	bool isBetweenPoints(const vector2d<T>& begin, const vector2d<T>& end) const
	{
		if (begin.X != end.X)
		{
			return ((begin.X <= X && X <= end.X) ||
				(begin.X >= X && X >= end.X));
		}
		else
		{
			return ((begin.Y <= Y && Y <= end.Y) ||
				(begin.Y >= Y && Y >= end.Y));
		}
	}

	//! Sets this vector to the linearly interpolated vector between a and b.
	/** \param a first vector to interpolate with, maximum at 1.0f
	\param b second vector to interpolate with, maximum at 0.0f
	\param d Interpolation value between 0.0f (all vector b) and 1.0f (all vector a)
	Note that this is the opposite direction of interpolation to getInterpolated_quadratic()
	*/
	vector2d<T>& interpolate(const vector2d<T>& a, const vector2d<T>& b, double d)
	{
		X = (T)((double)b.X + ( ( a.X - b.X ) * d ));
		Y = (T)((double)b.Y + ( ( a.Y - b.Y ) * d ));
		return *this;
	}

	//! X coordinate of vector.
	T X;

	//! Y coordinate of vector.
	T Y;
};

	//! Typedef for float 2d vector.
	typedef vector2d<float> vector2df;

	//! Typedef for integer 2d vector.
	typedef vector2d<int32_t> vector2di;

	template<class S, class T>
	vector2d<T> operator*(const S scalar, const vector2d<T>& vector) { return vector*scalar; }

	// These methods are declared in dimension2d, but need definitions of vector2d
	template<class T>
	dimension2d<T>::dimension2d(const vector2d<T>& other) : Width(other.X), Height(other.Y) { }

	template<class T>
	bool dimension2d<T>::operator==(const vector2d<T>& other) const { return Width == other.X && Height == other.Y; }

} // end namespace core
} // end namespace irr

#endif

