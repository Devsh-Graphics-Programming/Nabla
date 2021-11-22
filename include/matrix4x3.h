// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_MATRIX_4X3_H_INCLUDED_
#define _NBL_MATRIX_4X3_H_INCLUDED_

#include "vector3d.h"
#include "vectorSIMD.h"
#include "rect.h"


namespace nbl::core
{


	class matrix4x3// : public AlignedBase<_NBL_SIMD_ALIGNMENT> don't inherit from AlignedBase (which is empty) because member `rows[4]` inherits from it as well
	{
		public:
			//! Default constructor
			/** \param constructor Choose the initialization style */
			inline matrix4x3()
            {
                column[0].set(1.f,0.f,0.f);
                column[1].set(0.f,1.f,0.f);
                column[2].set(0.f,0.f,1.f);
                column[3].set(0.f,0.f,0.f);
            }
			//! Copy constructor
			/** \param other Other matrix to copy from
			\param constructor Choose the initialization style */
			inline matrix4x3(const matrix4x3& other)
			{
                *this = other;
			}

			//! Simple operator for directly accessing every element of the matrix.
			inline float& operator()(const size_t &i, const size_t &j) {return reinterpret_cast<float*>(column+j)[i];}

			//! Simple operator for directly accessing every element of the matrix.
			inline const float& operator()(const size_t& i, const size_t& j) const { return reinterpret_cast<const float*>(column+j)[i]; }

			inline vector3df& getColumn(const size_t& index) {return column[index];}
			inline const vector3df& getColumn(const size_t& index) const {return column[index];}

			//! Sets this matrix equal to the other matrix.
			inline matrix4x3& operator=(const matrix4x3 &other)
			{
			    column[0] = other.column[0];
			    column[1] = other.column[1];
			    column[2] = other.column[2];
			    column[3] = other.column[3];
			    return *this;
			}

			//! Returns pointer to internal array
			inline const float* pointer() const { return reinterpret_cast<const float*>(column); }
			inline float* pointer() { return reinterpret_cast<float*>(column); }

			//! Returns true if other matrix is equal to this matrix.
			inline bool operator==(const matrix4x3 &other) const
			{
			    return column[0]==other.column[0]&&column[1]==other.column[1]&&column[2]==other.column[2]&&column[3]==other.column[3];
			}

			//! Returns true if other matrix is not equal to this matrix.
			inline bool operator!=(const matrix4x3 &other) const
			{
			    return column[0]!=other.column[0]||column[1]!=other.column[1]||column[2]!=other.column[2]||column[3]!=other.column[3];
			}

			//! Add another matrix.
			inline matrix4x3 operator+(const matrix4x3& other) const
			{
			    matrix4x3 ret(*this);
			    return ret += other;
			}

			//! Add another matrix.
			inline matrix4x3& operator+=(const matrix4x3& other)
			{
			    column[0] += other.column[0];
			    column[1] += other.column[1];
			    column[2] += other.column[2];
			    column[3] += other.column[3];
			    return *this;
			}

			//! Subtract another matrix.
			inline matrix4x3 operator-(const matrix4x3& other) const
			{
			    matrix4x3 ret(*this);
			    return ret -= other;
			}

			//! Subtract another matrix.
			inline matrix4x3& operator-=(const matrix4x3& other)
			{
			    column[0] -= other.column[0];
			    column[1] -= other.column[1];
			    column[2] -= other.column[2];
			    column[3] -= other.column[3];
			    return *this;
			}

			//! Multiply by scalar.
			inline matrix4x3 operator*(const float& scalar) const
			{
			    matrix4x3 ret(*this);
			    return ret *= scalar;
			}

			//! Multiply by scalar.
			inline matrix4x3& operator*=(const float& scalar)
			{
			    column[0] *= scalar;
			    column[1] *= scalar;
			    column[2] *= scalar;
			    column[3] *= scalar;
			    return *this;
			}

			//! Set matrix to identity.
			inline matrix4x3& makeIdentity()
			{
			    column[0].set(1.f,0.f,0.f);
			    column[1].set(0.f,1.f,0.f);
			    column[2].set(0.f,0.f,1.f);
			    column[3].set(0.f,0.f,0.f);
			    return *this;
			}

			//! Returns true if the matrix is the identity matrix
			inline bool isIdentity() const
			{
			    return column[0]==vector3df(1.f,0.f,0.f)&&column[1]==vector3df(0.f,1.f,0.f)&&column[2]==vector3df(0.f,0.f,1.f)&&column[3]==vector3df(0.f,0.f,0.f);
			}

			//! Set the translation of the current matrix. Will erase any previous values.
			inline matrix4x3& setTranslation( const vector3df& translation )
			{
			    column[3] = translation;
			    return *this;
			}

			//! Gets the current translation
			inline const vector3df& getTranslation() const {return column[3];}

			//! Set the inverse translation of the current matrix. Will erase any previous values.
			inline matrix4x3& setInverseTranslation( const vector3df& translation )
			{
			    column[3] = -translation;
			    return *this;
			}

			//! Make a rotation matrix from Euler angles. The 4th row and column are unmodified.
			inline matrix4x3& setRotationRadians( const vector3df& rotation );

			//! Make a rotation matrix from Euler angles. The 4th row and column are unmodified.
			inline matrix4x3& setRotationDegrees( const vector3df& rotation )
            {
                return setRotationRadians( radians<vector3df>(rotation) );
            }

			//! Returns the rotation, as set by setRotation().
			/** This code was orginally written by by Chev. */
			inline core::vector3df getRotationDegrees() const;

			//! Make a rotation matrix from angle and axis, assuming left handed rotation.
			/** The 4th row and column are unmodified. */
			inline matrix4x3& setRotationAxisRadians(const float& angle, const vector3df& axis);

			//! Set Scale
			inline matrix4x3& setScale( const vector3df& scale );

			//! Set Scale
			inline matrix4x3& setScale( const float scale ) { return setScale(core::vector3df(scale,scale,scale)); }

			//! Get Scale
			inline core::vector3df getScale() const;

			//! Translate a vector by the translation part of this matrix.
			inline void translateVect( vector3df& vect ) const
			{
			    vect += column[3];
			}

			//! Translate a vector by the inverse of the translation part of this matrix.
			inline void inverseTranslateVect( vector3df& vect ) const
			{
			    vect -= column[3];
			}


			//! Sets all matrix data members at once
			inline matrix4x3& setM(const float* data)
			{
			    std::copy(data,data+12,reinterpret_cast<float*>(column));
				return *this;
			}


		private:
			//! Matrix data, stored in row-major order
			vector3df column[4];
	};


    inline matrix4x3 concatenateBFollowedByA(const matrix4x3& other_a,const matrix4x3& other_b )
    {
        matrix4x3 ret;

        ret.getColumn(0) = other_a.getColumn(0)*other_b(0,0)+other_a.getColumn(1)*other_b(1,0)+other_a.getColumn(2)*other_b(2,0);
        ret.getColumn(1) = other_a.getColumn(0)*other_b(0,1)+other_a.getColumn(1)*other_b(1,1)+other_a.getColumn(2)*other_b(2,1);
        ret.getColumn(2) = other_a.getColumn(0)*other_b(0,2)+other_a.getColumn(1)*other_b(1,2)+other_a.getColumn(2)*other_b(2,2);
        ret.getColumn(3) = other_a.getColumn(0)*other_b(0,3)+other_a.getColumn(1)*other_b(1,3)+other_a.getColumn(2)*other_b(2,3)+other_a.getColumn(3);

        return ret;
    }
    inline matrix4x3 concatenatePreciselyBFollowedByA(const matrix4x3& other_a,const matrix4x3& other_b )
    {
        matrix4x3 ret;
        core::vector3d<double> aColumns[4];
        core::vector3d<double> outColumn[4];
        for (size_t i=0; i<4; i++)
        for (size_t j=0; j<3; j++)
            reinterpret_cast<double*>(aColumns+i)[j] = double(reinterpret_cast<const float*>(&other_a.getColumn(i))[j]);

        outColumn[0] = aColumns[0]*double(other_b(0,0))+aColumns[1]*double(other_b(1,0))+aColumns[2]*double(other_b(2,0));
        outColumn[1] = aColumns[0]*double(other_b(0,1))+aColumns[1]*double(other_b(1,1))+aColumns[2]*double(other_b(2,1));
        outColumn[2] = aColumns[0]*double(other_b(0,2))+aColumns[1]*double(other_b(1,2))+aColumns[2]*double(other_b(2,2));
        outColumn[3] = aColumns[0]*double(other_b(0,3))+aColumns[1]*double(other_b(1,3))+aColumns[2]*double(other_b(2,3))+aColumns[3];

        for (size_t i=0; i<4; i++)
        {
            ret(0,i) = static_cast<float>(outColumn[i].X);
            ret(1,i) = static_cast<float>(outColumn[i].Y);
            ret(2,i) = static_cast<float>(outColumn[i].Z);
        }

        return ret;
    }



	inline matrix4x3& matrix4x3::setRotationRadians( const vector3df& rotation )
	{
		const float cr = cosf( rotation.X );
		const float sr = sinf( rotation.X );
		const float cp = cosf( rotation.Y );
		const float sp = sinf( rotation.Y );
		const float cy = cosf( rotation.Z );
		const float sy = sinf( rotation.Z );

		column[0].X = (float)( cp*cy );
		column[0].Y = (float)( cp*sy );
		column[0].Z = (float)( -sp );

		const float srsp = sr*sp;
		const float crsp = cr*sp;

		column[1].X = (float)( srsp*cy-cr*sy );
		column[1].Y = (float)( srsp*sy+cr*cy );
		column[1].Z = (float)( sr*cp );

		column[2].X = (float)( crsp*cy+sr*sy );
		column[2].Y = (float)( crsp*sy-sr*cy );
		column[2].Z = (float)( cr*cp );

		return *this;
	}


	//! Returns a rotation that is equivalent to that set by setRotationDegrees().
	/** This code was sent in by Chev.  Note that it does not necessarily return
	the *same* Euler angles as those set by setRotationDegrees(), but the rotation will
	be equivalent, i.e. will have the same result when used to rotate a vector or node. */
	inline core::vector3df matrix4x3::getRotationDegrees() const
	{
		const matrix4x3 &mat = *this;
		core::vector3df scale = getScale();

		const core::vector3d<float> invScale = core::reciprocal(scale);

		float nzd00 = mat(0,0)*invScale.X;
		float nzd11 = mat(1,1)*invScale.Y;
		float nzd22 = mat(2,2)*invScale.Z;
		//float trace = nzd00+nzd11+nzd22+1.f;

		float Y = -asinf(core::clamp(mat(2,0)*invScale.X, -1.f, 1.f));
		const float C = cosf(Y);
		Y = core::degrees(Y);

		float rotx, roty, X, Z;

		if (!core::iszero(C))
		{
			const float invC = core::reciprocal_approxim(C);
			rotx = nzd22 * invC;
			roty = mat(2,1) * invC * invScale.Y;
			X = core::degrees(atan2f( roty, rotx ));
			rotx = nzd00 * invC;
			roty = mat(1,0) * invC * invScale.X;
			Z = core::degrees(atan2f( roty, rotx ));
		}
		else
		{
			X = 0.0;
			roty = -mat(0,1) * invScale.Y;
			Z = core::degrees(atan2f( roty, nzd11 ));
		}

		// fix values that get below zero
		if (X < 0.0) X += 360.0;
		if (Y < 0.0) Y += 360.0;
		if (Z < 0.0) Z += 360.0;

		return vector3df((float)X,(float)Y,(float)Z);
	}

	//! Sets matrix to rotation matrix defined by axis and angle, assuming LH rotation

	inline matrix4x3& matrix4x3::setRotationAxisRadians( const float& angle, const vector3df& axis )
	{
 		const float c = cosf(angle);
		const float s = sinf(angle);
		const float t = 1.f - c;

		const float tx  = t * axis.X;
		const float ty  = t * axis.Y;
		const float tz  = t * axis.Z;

		const float sx  = s * axis.X;
		const float sy  = s * axis.Y;
		const float sz  = s * axis.Z;

		column[0].set(tx * axis.X + c,tx * axis.Y + sz,tx * axis.Z - sy);
        column[1].set(ty * axis.X - sz,ty * axis.Y + c,ty * axis.Z + sx);
        column[2].set(tz * axis.X + sy,tz * axis.Y - sx,tz * axis.Z + c);

		return *this;
	}


	inline matrix4x3& matrix4x3::setScale( const vector3df& scale )
	{
		column[0].X = scale.X;
		column[1].Y = scale.Y;
		column[2].Z = scale.Z;

		return *this;
	}

	//! Returns the absolute values of the scales of the matrix.
	/**
	Note that this returns the absolute (positive) values unless only scale is set.
	Unfortunately it does not appear to be possible to extract any original negative
	values. The best that we could do would be to arbitrarily make one scale
	negative if one or three of them were negative.
	FIXME - return the original values.
	*/

	inline vector3df matrix4x3::getScale() const
	{
		// See http://www.robertblum.com/articles/2005/02/14/decomposing-matrices
		// We have to do the full calculation.
		vector3df tmpScale(sqrtf(column[0].dotProduct(column[0])),sqrtf(column[1].dotProduct(column[1])),sqrtf(column[2].dotProduct(column[2])));
		if (column[0].dotProduct(column[1].crossProduct(column[2]))<0.f)
            tmpScale.Z = -tmpScale.Z;
        return tmpScale;
	}

} // end namespace nbl::core

#endif

