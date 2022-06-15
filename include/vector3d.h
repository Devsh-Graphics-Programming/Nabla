// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_POINT_3D_H_INCLUDED__
#define __NBL_POINT_3D_H_INCLUDED__

#include "nbl/core/math/glslFunctions.h"

namespace nbl
{
namespace core
{

	//! 3d vector template class with lots of operators and methods.
	/** The vector3d class is used in Irrlicht for three main purposes:
		1) As a direction vector (most of the methods assume this).
		2) As a position in 3d space (which is synonymous with a direction vector from the origin to this position).
		3) To hold three Euler rotations, where X is pitch, Y is yaw and Z is roll.
	*/
	template <class T>
	class NBL_API vector3d// : public AllocationOverrideDefault
	{
	public:
		//! Default constructor (null vector).
		vector3d() : X(0), Y(0), Z(0) {}
		//! Constructor with three different values
		vector3d(T nx, T ny, T nz) : X(nx), Y(ny), Z(nz) {}
		//! Constructor with the same value for all elements
		explicit vector3d(T n) : X(n), Y(n), Z(n) {}
		//! Copy constructor
		vector3d(const vector3d<T>& other) : X(other.X), Y(other.Y), Z(other.Z) {}
		//!
		vector3d(const T* pointerToArray) : X(pointerToArray[0]), Y(pointerToArray[1]), Z(pointerToArray[2]) {}

		// operators

		vector3d<T> operator-() const { return vector3d<T>(-X, -Y, -Z); }

		vector3d<T>& operator=(const vector3d<T>& other) { X = other.X; Y = other.Y; Z = other.Z; return *this; }

		vector3d<T> operator+(const vector3d<T>& other) const { return vector3d<T>(X + other.X, Y + other.Y, Z + other.Z); }
		vector3d<T>& operator+=(const vector3d<T>& other) { X+=other.X; Y+=other.Y; Z+=other.Z; return *this; }
		vector3d<T> operator+(const T val) const { return vector3d<T>(X + val, Y + val, Z + val); }
		vector3d<T>& operator+=(const T val) { X+=val; Y+=val; Z+=val; return *this; }

		vector3d<T> operator-(const vector3d<T>& other) const { return vector3d<T>(X - other.X, Y - other.Y, Z - other.Z); }
		vector3d<T>& operator-=(const vector3d<T>& other) { X-=other.X; Y-=other.Y; Z-=other.Z; return *this; }
		vector3d<T> operator-(const T val) const { return vector3d<T>(X - val, Y - val, Z - val); }
		vector3d<T>& operator-=(const T val) { X-=val; Y-=val; Z-=val; return *this; }

		vector3d<T> operator*(const vector3d<T>& other) const { return vector3d<T>(X * other.X, Y * other.Y, Z * other.Z); }
		vector3d<T>& operator*=(const vector3d<T>& other) { X*=other.X; Y*=other.Y; Z*=other.Z; return *this; }
		vector3d<T> operator*(const T v) const { return vector3d<T>(X * v, Y * v, Z * v); }
		vector3d<T>& operator*=(const T v) { X*=v; Y*=v; Z*=v; return *this; }

		vector3d<T> operator/(const vector3d<T>& other) const { return vector3d<T>(X / other.X, Y / other.Y, Z / other.Z); }
		vector3d<T>& operator/=(const vector3d<T>& other) { X/=other.X; Y/=other.Y; Z/=other.Z; return *this; }
		vector3d<T> operator/(const T v) const { T i=(T)1.0/v; return vector3d<T>(X * i, Y * i, Z * i); }
		vector3d<T>& operator/=(const T v) { T i=(T)1.0/v; X*=i; Y*=i; Z*=i; return *this; }

		//! use weak float compare
		bool operator==(const vector3d<T>& other) const
		{
			return core::equals<vector3d<T> >(*this,other,vector3d<T>(core::ROUNDING_ERROR<T>()));
		}

		bool operator!=(const vector3d<T>& other) const
		{
			return !operator==(other);
		}

		// functions

		vector3d<T>& set(const T nx, const T ny, const T nz) {X=nx; Y=ny; Z=nz; return *this;}
		vector3d<T>& set(const vector3d<T>& p) {X=p.X; Y=p.Y; Z=p.Z;return *this;}

		//! Get length of the vector.
		T getLength() const { return core::sqrt( X*X + Y*Y + Z*Z ); }

		//! Get squared length of the vector.
		/** This is useful because it is much faster than getLength().
		\return Squared length of the vector. */
		T getLengthSQ() const { return X*X + Y*Y + Z*Z; }

		//! Get the dot product with another vector.
		T dotProduct(const vector3d<T>& other) const
		{
			return X*other.X + Y*other.Y + Z*other.Z;
		}

		//! Get distance from another point.
		/** Here, the vector is interpreted as point in 3 dimensional space. */
		T getDistanceFrom(const vector3d<T>& other) const
		{
			return vector3d<T>(X - other.X, Y - other.Y, Z - other.Z).getLength();
		}

		//! Returns squared distance from another point.
		/** Here, the vector is interpreted as point in 3 dimensional space. */
		T getDistanceFromSQ(const vector3d<T>& other) const
		{
			return vector3d<T>(X - other.X, Y - other.Y, Z - other.Z).getLengthSQ();
		}

		//! Calculates the cross product with another vector.
		/** \param p Vector to multiply with.
		\return Crossproduct of this vector with p. */
		vector3d<T> crossProduct(const vector3d<T>& p) const
		{
			return vector3d<T>(Y * p.Z - Z * p.Y, Z * p.X - X * p.Z, X * p.Y - Y * p.X);
		}

		//! Returns if this vector interpreted as a point is on a line between two other points.
		/** It is assumed that the point is on the line.
		\param begin Beginning vector to compare between.
		\param end Ending vector to compare between.
		\return True if this vector is between begin and end, false if not. */
		bool isBetweenPoints(const vector3d<T>& begin, const vector3d<T>& end) const
		{
			const T f = (end - begin).getLengthSQ();
			return getDistanceFromSQ(begin) <= f &&
				getDistanceFromSQ(end) <= f;
		}

		//! Inverts the vector.
		vector3d<T>& invert()
		{
			X *= -1;
			Y *= -1;
			Z *= -1;
			return *this;
		}

		//! Sets this vector to the linearly interpolated vector between a and b.
		/** \param a first vector to interpolate with, maximum at 1.0f
		\param b second vector to interpolate with, maximum at 0.0f
		\param d Interpolation value between 0.0f (all vector b) and 1.0f (all vector a)
		Note that this is the opposite direction of interpolation to getInterpolated_quadratic()
		*/
		vector3d<T>& interpolate(const vector3d<T>& a, const vector3d<T>& b, double d)
		{
			X = (T)((double)b.X + ( ( a.X - b.X ) * d ));
			Y = (T)((double)b.Y + ( ( a.Y - b.Y ) * d ));
			Z = (T)((double)b.Z + ( ( a.Z - b.Z ) * d ));
			return *this;
		}


		//! Get the rotations that would make a (0,0,1) direction vector point in the same direction as this direction vector.
		/** Thanks to Arras on the Irrlicht forums for this method.  This utility method is very useful for
		orienting scene nodes towards specific targets.  For example, if this vector represents the difference
		between two scene nodes, then applying the result of getHorizontalAngle() to one scene node will point
		it at the other one.
		Example code:
		// Where target and seeker are of type ISceneNode*
		const vector3df toTarget(target->getAbsolutePosition() - seeker->getAbsolutePosition());
		const vector3df requiredRotation = toTarget.getHorizontalAngle();
		seeker->setRotation(requiredRotation);

		\return A rotation vector containing the X (pitch) and Y (raw) rotations (in degrees) that when applied to a
		+Z (e.g. 0, 0, 1) direction vector would make it point in the same direction as this vector. The Z (roll) rotation
		is always 0, since two Euler rotations are sufficient to point in any given direction. */
		vector3d<T> getHorizontalAngle() const
		{
			vector3d<T> angle;

			const double tmp = core::degrees(atan2((double)X, (double)Z));
			angle.Y = (T)tmp;

			if (angle.Y < 0)
				angle.Y += 360;
			if (angle.Y >= 360)
				angle.Y -= 360;

			const double z1 = core::sqrt(X*X + Z*Z);

			angle.X = (T)(core::degrees(atan2((double)z1, (double)Y)) - 90.0);

			if (angle.X < 0)
				angle.X += 360;
			if (angle.X >= 360)
				angle.X -= 360;

			return angle;
		}

		//! Builds a direction vector from (this) rotation vector.
		/** This vector is assumed to be a rotation vector composed of 3 Euler angle rotations, in degrees.
		The implementation performs the same calculations as using a matrix to do the rotation.

		\param[in] forwards  The direction representing "forwards" which will be rotated by this vector.
		If you do not provide a direction, then the +Z axis (0, 0, 1) will be assumed to be forwards.
		\return A direction vector calculated by rotating the forwards direction by the 3 Euler angles
		(in degrees) represented by this vector. */
		vector3d<T> rotationToDirection(const vector3d<T> & forwards = vector3d<T>(0, 0, 1)) const
		{
			const double cr = cos( core::radians(X) );
			const double sr = sin( core::radians(X) );
			const double cp = cos( core::radians(Y) );
			const double sp = sin( core::radians(Y) );
			const double cy = cos( core::radians(Z) );
			const double sy = sin( core::radians(Z) );

			const double srsp = sr*sp;
			const double crsp = cr*sp;

			const double pseudoMatrix[] = {
				( cp*cy ), ( cp*sy ), ( -sp ),
				( srsp*cy-cr*sy ), ( srsp*sy+cr*cy ), ( sr*cp ),
				( crsp*cy+sr*sy ), ( crsp*sy-sr*cy ), ( cr*cp )};

			return vector3d<T>(
				(T)(forwards.X * pseudoMatrix[0] +
					forwards.Y * pseudoMatrix[3] +
					forwards.Z * pseudoMatrix[6]),
				(T)(forwards.X * pseudoMatrix[1] +
					forwards.Y * pseudoMatrix[4] +
					forwards.Z * pseudoMatrix[7]),
				(T)(forwards.X * pseudoMatrix[2] +
					forwards.Y * pseudoMatrix[5] +
					forwards.Z * pseudoMatrix[8]));
		}

		//! Fills an array of 4 values with the vector data (usually floats).
		/** Useful for setting in shader constants for example. The fourth value
		will always be 0. */
		void getAs4Values(T* array) const
		{
			array[0] = X;
			array[1] = Y;
			array[2] = Z;
			array[3] = 0;
		}

		//! Fills an array of 3 values with the vector data (usually floats).
		/** Useful for setting in shader constants for example.*/
		void getAs3Values(T* array) const
		{
			array[0] = X;
			array[1] = Y;
			array[2] = Z;
		}


		//! X coordinate of the vector
		T X;

		//! Y coordinate of the vector
		T Y;

		//! Z coordinate of the vector
		T Z;
	};

	//! partial specialization for integer vectors
	// Implementor note: inline keyword needed due to template specialization for int32_t. Otherwise put specialization into a .cpp
	template <>
	inline vector3d<int32_t> vector3d<int32_t>::operator /(int32_t val) const {return core::vector3d<int32_t>(X/val,Y/val,Z/val);}
	template <>
	inline vector3d<int32_t>& vector3d<int32_t>::operator /=(int32_t val) {X/=val;Y/=val;Z/=val; return *this;}


	//! Typedef for a float 3d vector.
	typedef vector3d<float> vector3df;

	//! Typedef for an integer 3d vector.
	typedef vector3d<int32_t> vector3di;

	//! Typedef for an integer 3d vector.
	typedef vector3d<int32_t> vector3dint32_t;	// sodan

	//! Function multiplying a scalar and a vector component-wise.
	template<class S, class T>
	vector3d<T> operator*(const S scalar, const vector3d<T>& vector) { return vector*scalar; }


} // end namespace core
} // end namespace nbl

#endif

