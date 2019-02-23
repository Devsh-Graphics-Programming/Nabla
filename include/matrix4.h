// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MATRIX_H_INCLUDED__
#define __IRR_MATRIX_H_INCLUDED__

#include "vector3d.h"
#include "vector2d.h"
#include "plane3d.h"
#include "aabbox3d.h"
#include "rect.h"
#include "string.h"

// enable this to keep track of changes to the matrix
// and make simpler identity check for seldomly changing matrices
// otherwise identity check will always compare the elements
//#define USE_MATRIX_TEST

// this is only for debugging purposes
//#define USE_MATRIX_TEST_DEBUG

#if defined( USE_MATRIX_TEST_DEBUG )

struct MatrixTest
{
	MatrixTest () : ID(0), Calls(0) {}
	char buf[256];
	int Calls;
	int ID;
};
static MatrixTest MTest;

#endif

namespace irr
{
namespace core
{

	//! 4x4 matrix. Mostly used as transformation matrix for 3d calculations.
	/** The matrix is a D3D style matrix, row major with translations in the 4th row. */
	template <class T>
	class CMatrix4// : public AllocationOverrideDefault
	{
		public:

			//! Constructor Flags
			enum eConstructor
			{
				EM4CONST_NOTHING = 0,
				EM4CONST_COPY,
				EM4CONST_IDENTITY,
				EM4CONST_TRANSPOSED,
				EM4CONST_INVERSE,
				EM4CONST_INVERSE_TRANSPOSED
			};

			//! Default constructor
			/** \param constructor Choose the initialization style */
			CMatrix4( eConstructor constructor = EM4CONST_IDENTITY );
			//! Copy constructor
			/** \param other Other matrix to copy from
			\param constructor Choose the initialization style */
			CMatrix4(const CMatrix4<T>& other, eConstructor constructor = EM4CONST_COPY);

			//! Simple operator for directly accessing every element of the matrix.
			T& operator()(const int32_t row, const int32_t col)
			{
#if defined ( USE_MATRIX_TEST )
				definitelyIdentityMatrix=false;
#endif
				return M[ row * 4 + col ];
			}

			//! Simple operator for directly accessing every element of the matrix.
			const T& operator()(const int32_t row, const int32_t col) const { return M[row * 4 + col]; }

			//! Simple operator for linearly accessing every element of the matrix.
			T& operator[](uint32_t index)
			{
#if defined ( USE_MATRIX_TEST )
				definitelyIdentityMatrix=false;
#endif
				return M[index];
			}

			//! Simple operator for linearly accessing every element of the matrix.
			const T& operator[](uint32_t index) const { return M[index]; }

			//! Sets this matrix equal to the other matrix.
			inline CMatrix4<T>& operator=(const CMatrix4<T> &other);

			//! Sets all elements of this matrix to the value.
			inline CMatrix4<T>& operator=(const T& scalar);

			//! Returns pointer to internal array
			const T* pointer() const { return M; }
			T* pointer()
			{
#if defined ( USE_MATRIX_TEST )
				definitelyIdentityMatrix=false;
#endif
				return M;
			}

			//! Returns true if other matrix is equal to this matrix.
			bool operator==(const CMatrix4<T> &other) const;

			//! Returns true if other matrix is not equal to this matrix.
			bool operator!=(const CMatrix4<T> &other) const;

			//! Add another matrix.
			CMatrix4<T> operator+(const CMatrix4<T>& other) const;

			//! Add another matrix.
			CMatrix4<T>& operator+=(const CMatrix4<T>& other);

			//! Subtract another matrix.
			CMatrix4<T> operator-(const CMatrix4<T>& other) const;

			//! Subtract another matrix.
			CMatrix4<T>& operator-=(const CMatrix4<T>& other);

			//! set this matrix to the product of two matrices
			/** Calculate b*a */
			inline CMatrix4<T>& setbyproduct(const CMatrix4<T>& other_a,const CMatrix4<T>& other_b );

			//! Set this matrix to the product of two matrices
			/** Calculate b*a, no optimization used,
			use it if you know you never have a identity matrix */
			CMatrix4<T>& setbyproduct_nocheck(const CMatrix4<T>& other_a,const CMatrix4<T>& other_b );

			//! Multiply by another matrix.
			/** Calculate other*this */
			CMatrix4<T> operator*(const CMatrix4<T>& other) const;

			//! Multiply by another matrix.
			/** Calculate and return other*this */
			CMatrix4<T>& operator*=(const CMatrix4<T>& other);

			//! Multiply by scalar.
			CMatrix4<T> operator*(const T& scalar) const;

			//! Multiply by scalar.
			CMatrix4<T>& operator*=(const T& scalar);

			//! Set matrix to identity.
			inline CMatrix4<T>& makeIdentity();

			//! Returns true if the matrix is the identity matrix
			inline bool isIdentity() const;

			//! Returns true if the matrix is orthogonal
			inline bool isOrthogonal() const;

			//! Returns true if the matrix is the identity matrix
			bool isIdentity_integer_base () const;

			//! Set the translation of the current matrix. Will erase any previous values.
			CMatrix4<T>& setTranslation( const vector3d<T>& translation );

			//! Gets the current translation
			vector3d<T> getTranslation() const;

			//! Set the inverse translation of the current matrix. Will erase any previous values.
			CMatrix4<T>& setInverseTranslation( const vector3d<T>& translation );

			//! Make a rotation matrix from Euler angles. The 4th row and column are unmodified.
			inline CMatrix4<T>& setRotationRadians( const vector3d<T>& rotation );

			//! Make a rotation matrix from Euler angles. The 4th row and column are unmodified.
			CMatrix4<T>& setRotationDegrees( const vector3d<T>& rotation );

			//! Returns the rotation, as set by setRotation().
			/** This code was orginally written by by Chev. */
			core::vector3d<T> getRotationDegrees() const;

			//! Make an inverted rotation matrix from Euler angles.
			/** The 4th row and column are unmodified. */
			inline CMatrix4<T>& setInverseRotationRadians( const vector3d<T>& rotation );

			//! Make an inverted rotation matrix from Euler angles.
			/** The 4th row and column are unmodified. */
			inline CMatrix4<T>& setInverseRotationDegrees( const vector3d<T>& rotation );

			//! Make a rotation matrix from angle and axis, assuming left handed rotation.
			/** The 4th row and column are unmodified. */
			inline CMatrix4<T>& setRotationAxisRadians(const T& angle, const vector3d<T>& axis);

			//! Set Scale
			CMatrix4<T>& setScale( const vector3d<T>& scale );

			//! Set Scale
			CMatrix4<T>& setScale( const T scale ) { return setScale(core::vector3d<T>(scale,scale,scale)); }

			//! Get Scale
			core::vector3d<T> getScale() const;

			//! Translate a vector by the inverse of the translation part of this matrix.
			void inverseTranslateVect( vector3df& vect ) const;

			//! Rotate a vector by the rotation part of this matrix.
			void rotateVect( vector3df& vect ) const;

			//! An alternate transform vector method, writing into a second vector
			void rotateVect(core::vector3df& out, const core::vector3df& in) const;

			//! An alternate transform vector method, writing into an array of 3 floats
			void rotateVect(T *out,const core::vector3df &in) const;

			//! Transforms the vector by this matrix
			void transformVect( vector3df& vect) const;

			//! Transforms input vector by this matrix and stores result in output vector
			void transformVect( vector3df& out, const vector3df& in ) const;

			//! An alternate transform vector method, writing into an array of 4 floats
			void transformVect(T *out,const core::vector3df &in) const;

			//! An alternate transform vector method, reading from and writing to an array of 3 floats
			void transformVec3(T *out, const T * in) const;

			//! Translate a vector by the translation part of this matrix.
			void translateVect( vector3df& vect ) const;

			//! Transforms a plane by this matrix
			void transformPlane( core::plane3d<float> &plane) const;

			//! Transforms a plane by this matrix
			void transformPlane( const core::plane3d<float> &in, core::plane3d<float> &out) const;

			//! Multiplies this matrix by a 1x4 matrix
			void multiplyWith1x4Matrix(T* matrix) const;

			//! Calculates inverse of matrix. Slow.
			/** \return Returns false if there is no inverse matrix.*/
			bool makeInverse();


			//! Inverts a primitive matrix which only contains a translation and a rotation
			/** \param out: where result matrix is written to. */
			bool getInversePrimitive ( CMatrix4<T>& out ) const;

			//! Gets the inversed matrix of this one
			/** \param out: where result matrix is written to.
			\return Returns false if there is no inverse matrix. */
			bool getInverse(CMatrix4<T>& out) const;

			//! Builds a right-handed perspective projection matrix based on a field of view
			CMatrix4<T>& buildProjectionMatrixPerspectiveFovRH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar);

			//! Builds a left-handed perspective projection matrix based on a field of view
			CMatrix4<T>& buildProjectionMatrixPerspectiveFovLH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar);

			//! Builds a left-handed orthogonal projection matrix.
			CMatrix4<T>& buildProjectionMatrixOrthoLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar);

			//! Builds a right-handed orthogonal projection matrix.
			CMatrix4<T>& buildProjectionMatrixOrthoRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar);

			//! Builds a left-handed look-at matrix.
			CMatrix4<T>& buildCameraLookAtMatrixLH(
					const vector3df& position,
					const vector3df& target,
					const vector3df& upVector);

			//! Builds a right-handed look-at matrix.
			CMatrix4<T>& buildCameraLookAtMatrixRH(
					const vector3df& position,
					const vector3df& target,
					const vector3df& upVector);

			//! Creates a new matrix as interpolated matrix from two other ones.
			/** \param b: other matrix to interpolate with
			\param time: Must be a value between 0 and 1. */
			CMatrix4<T> interpolate(const core::CMatrix4<T>& b, float time) const;

			//! Gets transposed matrix
			CMatrix4<T> getTransposed() const;

			//! Gets transposed matrix
			inline void getTransposed( CMatrix4<T>& dest ) const;

			//! Builds a combined matrix which translates to a center before rotation and translates from origin afterwards
			/** \param center Position to rotate around
			\param translate Translation applied after the rotation
			 */
			void setRotationCenter(const core::vector3df& center, const core::vector3df& translate);

			//! Set texture transformation rotation
			/** Rotate about z axis, recenter at (0.5,0.5).
			Doesn't clear other elements than those affected
			\param radAngle Angle in radians
			\return Altered matrix */
			CMatrix4<T>& setTextureRotationCenter( float radAngle );

			//! Set texture transformation translation
			/** Doesn't clear other elements than those affected.
			\param x Offset on x axis
			\param y Offset on y axis
			\return Altered matrix */
			CMatrix4<T>& setTextureTranslate( float x, float y );

			//! Set texture transformation translation, using a transposed representation
			/** Doesn't clear other elements than those affected.
			\param x Offset on x axis
			\param y Offset on y axis
			\return Altered matrix */
			CMatrix4<T>& setTextureTranslateTransposed( float x, float y );

			//! Set texture transformation scale
			/** Doesn't clear other elements than those affected.
			\param sx Scale factor on x axis
			\param sy Scale factor on y axis
			\return Altered matrix. */
			CMatrix4<T>& setTextureScale( float sx, float sy );

			//! Set texture transformation scale, and recenter at (0.5,0.5)
			/** Doesn't clear other elements than those affected.
			\param sx Scale factor on x axis
			\param sy Scale factor on y axis
			\return Altered matrix. */
			CMatrix4<T>& setTextureScaleCenter( float sx, float sy );

			//! Sets all matrix data members at once
			CMatrix4<T>& setM(const T* data);

			//! Sets if the matrix is definitely identity matrix
			void setDefinitelyIdentityMatrix( bool isDefinitelyIdentityMatrix);

			//! Gets if the matrix is definitely identity matrix
			bool getDefinitelyIdentityMatrix() const;

			//! Compare two matrices using the equal method
			bool equals(const core::CMatrix4<T>& other, const T tolerance=(T)ROUNDING_ERROR_f64) const;

			inline bool isBoxInsideFrustum(const core::aabbox3df& tbox)
			{
                core::vector3df column0_Min = reinterpret_cast<core::vector3df*>(M)[0];
                core::vector3df column1_Min = reinterpret_cast<core::vector3df*>(M+4)[0];
                core::vector3df column2_Min = reinterpret_cast<core::vector3df*>(M+8)[0];
                core::vector3df column3_Min = reinterpret_cast<core::vector3df*>(M+12)[0];
                core::vector3df column0_Max = column0_Min;
                core::vector3df column1_Max = column1_Min;
                core::vector3df column2_Max = column2_Min;
                core::vector3df column3_Max = column3_Min;

                column0_Min += M[3];
                column1_Min += M[7];
                column2_Min += M[11];
                column3_Min += M[15];

                float tmp = column0_Min.X*(column0_Min.X<0.f ? tbox.MinEdge.X:tbox.MaxEdge.X)+column1_Min.X*(column1_Min.X<0.f ? tbox.MinEdge.Y:tbox.MaxEdge.Y)+column2_Min.X*(column2_Min.X<0.f ? tbox.MinEdge.Z:tbox.MaxEdge.Z);
                if (tmp<=-column3_Min.X)
                    return false;
                tmp = column0_Min.Y*(column0_Min.Y<0.f ? tbox.MinEdge.X:tbox.MaxEdge.X)+column1_Min.Y*(column1_Min.Y<0.f ? tbox.MinEdge.Y:tbox.MaxEdge.Y)+column2_Min.Y*(column2_Min.Y<0.f ? tbox.MinEdge.Z:tbox.MaxEdge.Z);
                if (tmp<=-column3_Min.Y)
                    return false;
                tmp = M[3]*(M[3]<0.f ? tbox.MinEdge.X:tbox.MaxEdge.X)+M[7]*(M[7]<0.f ? tbox.MinEdge.Y:tbox.MaxEdge.Y)+M[11]*(M[11]<0.f ? tbox.MinEdge.Z:tbox.MaxEdge.Z);
                if (tmp<=-M[15])
                    return false;

                column0_Max -= M[3];
                column1_Max -= M[7];
                column2_Max -= M[11];
                column3_Max = core::vector3df(M[15])-column3_Max;
                tmp = column0_Max.X*(column0_Max.X>=0.f ? tbox.MinEdge.X:tbox.MaxEdge.X)+column1_Max.X*(column1_Max.X>=0.f ? tbox.MinEdge.Y:tbox.MaxEdge.Y)+column2_Max.X*(column2_Max.X>=0.f ? tbox.MinEdge.Z:tbox.MaxEdge.Z);
                if (tmp>=column3_Max.X)
                    return false;
                tmp = column0_Max.Y*(column0_Max.Y>=0.f ? tbox.MinEdge.X:tbox.MaxEdge.X)+column1_Max.Y*(column1_Max.Y>=0.f ? tbox.MinEdge.Y:tbox.MaxEdge.Y)+column2_Max.Y*(column2_Max.Y>=0.f ? tbox.MinEdge.Z:tbox.MaxEdge.Z);
                if (tmp>=column3_Max.Y)
                    return false;

                tmp = column0_Max.Z*(column0_Max.Z>=0.f ? tbox.MinEdge.X:tbox.MaxEdge.X)+column1_Max.Z*(column1_Max.Z>=0.f ? tbox.MinEdge.Y:tbox.MaxEdge.Y)+column2_Max.Z*(column2_Max.Z>=0.f ? tbox.MinEdge.Z:tbox.MaxEdge.Z);
                if (tmp>=column3_Max.Z)
                    return false;

                return true;
			}

		private:
			//! Matrix data, stored in row-major order
			T M[16];
#if defined ( USE_MATRIX_TEST )
			//! Flag is this matrix is identity matrix
			mutable uint32_t definitelyIdentityMatrix;
#endif
#if defined ( USE_MATRIX_TEST_DEBUG )
			uint32_t id;
			mutable uint32_t calls;
#endif

	};

	// Default constructor
	template <class T>
	inline CMatrix4<T>::CMatrix4( eConstructor constructor )
#if defined ( USE_MATRIX_TEST )
		: definitelyIdentityMatrix(BIT_UNTESTED)
#endif
#if defined ( USE_MATRIX_TEST_DEBUG )
		,id ( MTest.ID++), calls ( 0 )
#endif
	{
		switch ( constructor )
		{
			case EM4CONST_NOTHING:
			case EM4CONST_COPY:
				break;
			case EM4CONST_IDENTITY:
			case EM4CONST_INVERSE:
			default:
				makeIdentity();
				break;
		}
	}

	// Copy constructor
	template <class T>
	inline CMatrix4<T>::CMatrix4( const CMatrix4<T>& other, eConstructor constructor)
#if defined ( USE_MATRIX_TEST )
		: definitelyIdentityMatrix(BIT_UNTESTED)
#endif
#if defined ( USE_MATRIX_TEST_DEBUG )
		,id ( MTest.ID++), calls ( 0 )
#endif
	{
		switch ( constructor )
		{
			case EM4CONST_IDENTITY:
				makeIdentity();
				break;
			case EM4CONST_NOTHING:
				break;
			case EM4CONST_COPY:
				*this = other;
				break;
			case EM4CONST_TRANSPOSED:
				other.getTransposed(*this);
				break;
			case EM4CONST_INVERSE:
				if (!other.getInverse(*this))
					memset(M, 0, 16*sizeof(T));
				break;
			case EM4CONST_INVERSE_TRANSPOSED:
				if (!other.getInverse(*this))
					memset(M, 0, 16*sizeof(T));
				else
					*this=getTransposed();
				break;
		}
	}

	//! Add another matrix.
	template <class T>
	inline CMatrix4<T> CMatrix4<T>::operator+(const CMatrix4<T>& other) const
	{
		CMatrix4<T> temp ( EM4CONST_NOTHING );

		temp[0] = M[0]+other[0];
		temp[1] = M[1]+other[1];
		temp[2] = M[2]+other[2];
		temp[3] = M[3]+other[3];
		temp[4] = M[4]+other[4];
		temp[5] = M[5]+other[5];
		temp[6] = M[6]+other[6];
		temp[7] = M[7]+other[7];
		temp[8] = M[8]+other[8];
		temp[9] = M[9]+other[9];
		temp[10] = M[10]+other[10];
		temp[11] = M[11]+other[11];
		temp[12] = M[12]+other[12];
		temp[13] = M[13]+other[13];
		temp[14] = M[14]+other[14];
		temp[15] = M[15]+other[15];

		return temp;
	}

	//! Add another matrix.
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::operator+=(const CMatrix4<T>& other)
	{
		M[0]+=other[0];
		M[1]+=other[1];
		M[2]+=other[2];
		M[3]+=other[3];
		M[4]+=other[4];
		M[5]+=other[5];
		M[6]+=other[6];
		M[7]+=other[7];
		M[8]+=other[8];
		M[9]+=other[9];
		M[10]+=other[10];
		M[11]+=other[11];
		M[12]+=other[12];
		M[13]+=other[13];
		M[14]+=other[14];
		M[15]+=other[15];

		return *this;
	}

	//! Subtract another matrix.
	template <class T>
	inline CMatrix4<T> CMatrix4<T>::operator-(const CMatrix4<T>& other) const
	{
		CMatrix4<T> temp ( EM4CONST_NOTHING );

		temp[0] = M[0]-other[0];
		temp[1] = M[1]-other[1];
		temp[2] = M[2]-other[2];
		temp[3] = M[3]-other[3];
		temp[4] = M[4]-other[4];
		temp[5] = M[5]-other[5];
		temp[6] = M[6]-other[6];
		temp[7] = M[7]-other[7];
		temp[8] = M[8]-other[8];
		temp[9] = M[9]-other[9];
		temp[10] = M[10]-other[10];
		temp[11] = M[11]-other[11];
		temp[12] = M[12]-other[12];
		temp[13] = M[13]-other[13];
		temp[14] = M[14]-other[14];
		temp[15] = M[15]-other[15];

		return temp;
	}

	//! Subtract another matrix.
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::operator-=(const CMatrix4<T>& other)
	{
		M[0]-=other[0];
		M[1]-=other[1];
		M[2]-=other[2];
		M[3]-=other[3];
		M[4]-=other[4];
		M[5]-=other[5];
		M[6]-=other[6];
		M[7]-=other[7];
		M[8]-=other[8];
		M[9]-=other[9];
		M[10]-=other[10];
		M[11]-=other[11];
		M[12]-=other[12];
		M[13]-=other[13];
		M[14]-=other[14];
		M[15]-=other[15];

		return *this;
	}

	//! Multiply by scalar.
	template <class T>
	inline CMatrix4<T> CMatrix4<T>::operator*(const T& scalar) const
	{
		CMatrix4<T> temp ( EM4CONST_NOTHING );

		temp[0] = M[0]*scalar;
		temp[1] = M[1]*scalar;
		temp[2] = M[2]*scalar;
		temp[3] = M[3]*scalar;
		temp[4] = M[4]*scalar;
		temp[5] = M[5]*scalar;
		temp[6] = M[6]*scalar;
		temp[7] = M[7]*scalar;
		temp[8] = M[8]*scalar;
		temp[9] = M[9]*scalar;
		temp[10] = M[10]*scalar;
		temp[11] = M[11]*scalar;
		temp[12] = M[12]*scalar;
		temp[13] = M[13]*scalar;
		temp[14] = M[14]*scalar;
		temp[15] = M[15]*scalar;

		return temp;
	}

	//! Multiply by scalar.
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::operator*=(const T& scalar)
	{
		M[0]*=scalar;
		M[1]*=scalar;
		M[2]*=scalar;
		M[3]*=scalar;
		M[4]*=scalar;
		M[5]*=scalar;
		M[6]*=scalar;
		M[7]*=scalar;
		M[8]*=scalar;
		M[9]*=scalar;
		M[10]*=scalar;
		M[11]*=scalar;
		M[12]*=scalar;
		M[13]*=scalar;
		M[14]*=scalar;
		M[15]*=scalar;

		return *this;
	}

	//! Multiply by another matrix.
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::operator*=(const CMatrix4<T>& other)
	{
#if defined ( USE_MATRIX_TEST )
		// do checks on your own in order to avoid copy creation
		if ( !other.isIdentity() )
		{
			if ( this->isIdentity() )
			{
				return (*this = other);
			}
			else
			{
				CMatrix4<T> temp ( *this );
				return setbyproduct_nocheck( temp, other );
			}
		}
		return *this;
#else
		CMatrix4<T> temp ( *this );
		return setbyproduct_nocheck( temp, other );
#endif
	}

	//! multiply by another matrix
	// set this matrix to the product of two other matrices
	// goal is to reduce stack use and copy
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setbyproduct_nocheck(const CMatrix4<T>& other_a,const CMatrix4<T>& other_b )
	{
		const T *m1 = other_a.M;
		const T *m2 = other_b.M;

		M[0] = m1[0]*m2[0] + m1[4]*m2[1] + m1[8]*m2[2] + m1[12]*m2[3];
		M[1] = m1[1]*m2[0] + m1[5]*m2[1] + m1[9]*m2[2] + m1[13]*m2[3];
		M[2] = m1[2]*m2[0] + m1[6]*m2[1] + m1[10]*m2[2] + m1[14]*m2[3];
		M[3] = m1[3]*m2[0] + m1[7]*m2[1] + m1[11]*m2[2] + m1[15]*m2[3];

		M[4] = m1[0]*m2[4] + m1[4]*m2[5] + m1[8]*m2[6] + m1[12]*m2[7];
		M[5] = m1[1]*m2[4] + m1[5]*m2[5] + m1[9]*m2[6] + m1[13]*m2[7];
		M[6] = m1[2]*m2[4] + m1[6]*m2[5] + m1[10]*m2[6] + m1[14]*m2[7];
		M[7] = m1[3]*m2[4] + m1[7]*m2[5] + m1[11]*m2[6] + m1[15]*m2[7];

		M[8] = m1[0]*m2[8] + m1[4]*m2[9] + m1[8]*m2[10] + m1[12]*m2[11];
		M[9] = m1[1]*m2[8] + m1[5]*m2[9] + m1[9]*m2[10] + m1[13]*m2[11];
		M[10] = m1[2]*m2[8] + m1[6]*m2[9] + m1[10]*m2[10] + m1[14]*m2[11];
		M[11] = m1[3]*m2[8] + m1[7]*m2[9] + m1[11]*m2[10] + m1[15]*m2[11];

		M[12] = m1[0]*m2[12] + m1[4]*m2[13] + m1[8]*m2[14] + m1[12]*m2[15];
		M[13] = m1[1]*m2[12] + m1[5]*m2[13] + m1[9]*m2[14] + m1[13]*m2[15];
		M[14] = m1[2]*m2[12] + m1[6]*m2[13] + m1[10]*m2[14] + m1[14]*m2[15];
		M[15] = m1[3]*m2[12] + m1[7]*m2[13] + m1[11]*m2[14] + m1[15]*m2[15];
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	//! multiply by another matrix
	// set this matrix to the product of two other matrices
	// goal is to reduce stack use and copy
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setbyproduct(const CMatrix4<T>& other_a, const CMatrix4<T>& other_b )
	{
#if defined ( USE_MATRIX_TEST )
		if ( other_a.isIdentity () )
			return (*this = other_b);
		else
		if ( other_b.isIdentity () )
			return (*this = other_a);
		else
			return setbyproduct_nocheck(other_a,other_b);
#else
		return setbyproduct_nocheck(other_a,other_b);
#endif
	}

	//! multiply by another matrix
	template <class T>
	inline CMatrix4<T> CMatrix4<T>::operator*(const CMatrix4<T>& m2) const
	{
#if defined ( USE_MATRIX_TEST )
		// Testing purpose..
		if ( this->isIdentity() )
			return m2;
		if ( m2.isIdentity() )
			return *this;
#endif

		CMatrix4<T> m3 ( EM4CONST_NOTHING );

		const T *m1 = M;

		m3[0] = m1[0]*m2[0] + m1[4]*m2[1] + m1[8]*m2[2] + m1[12]*m2[3];
		m3[1] = m1[1]*m2[0] + m1[5]*m2[1] + m1[9]*m2[2] + m1[13]*m2[3];
		m3[2] = m1[2]*m2[0] + m1[6]*m2[1] + m1[10]*m2[2] + m1[14]*m2[3];
		m3[3] = m1[3]*m2[0] + m1[7]*m2[1] + m1[11]*m2[2] + m1[15]*m2[3];

		m3[4] = m1[0]*m2[4] + m1[4]*m2[5] + m1[8]*m2[6] + m1[12]*m2[7];
		m3[5] = m1[1]*m2[4] + m1[5]*m2[5] + m1[9]*m2[6] + m1[13]*m2[7];
		m3[6] = m1[2]*m2[4] + m1[6]*m2[5] + m1[10]*m2[6] + m1[14]*m2[7];
		m3[7] = m1[3]*m2[4] + m1[7]*m2[5] + m1[11]*m2[6] + m1[15]*m2[7];

		m3[8] = m1[0]*m2[8] + m1[4]*m2[9] + m1[8]*m2[10] + m1[12]*m2[11];
		m3[9] = m1[1]*m2[8] + m1[5]*m2[9] + m1[9]*m2[10] + m1[13]*m2[11];
		m3[10] = m1[2]*m2[8] + m1[6]*m2[9] + m1[10]*m2[10] + m1[14]*m2[11];
		m3[11] = m1[3]*m2[8] + m1[7]*m2[9] + m1[11]*m2[10] + m1[15]*m2[11];

		m3[12] = m1[0]*m2[12] + m1[4]*m2[13] + m1[8]*m2[14] + m1[12]*m2[15];
		m3[13] = m1[1]*m2[12] + m1[5]*m2[13] + m1[9]*m2[14] + m1[13]*m2[15];
		m3[14] = m1[2]*m2[12] + m1[6]*m2[13] + m1[10]*m2[14] + m1[14]*m2[15];
		m3[15] = m1[3]*m2[12] + m1[7]*m2[13] + m1[11]*m2[14] + m1[15]*m2[15];
		return m3;
	}



	template <class T>
	inline vector3d<T> CMatrix4<T>::getTranslation() const
	{
		return vector3d<T>(M[12], M[13], M[14]);
	}


	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setTranslation( const vector3d<T>& translation )
	{
		M[12] = translation.X;
		M[13] = translation.Y;
		M[14] = translation.Z;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}

	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setInverseTranslation( const vector3d<T>& translation )
	{
		M[12] = -translation.X;
		M[13] = -translation.Y;
		M[14] = -translation.Z;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}

	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setScale( const vector3d<T>& scale )
	{
		M[0] = scale.X;
		M[5] = scale.Y;
		M[10] = scale.Z;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
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
	template <class T>
	inline vector3d<T> CMatrix4<T>::getScale() const
	{
		// See http://www.robertblum.com/articles/2005/02/14/decomposing-matrices

		// Deal with the 0 rotation case first
		// Prior to Irrlicht 1.6, we always returned this value.
		if(core::iszero(M[1]) && core::iszero(M[2]) &&
			core::iszero(M[4]) && core::iszero(M[6]) &&
			core::iszero(M[8]) && core::iszero(M[9]))
			return vector3d<T>(M[0], M[5], M[10]);

		// We have to do the full calculation.
		return vector3d<T>(sqrtf(M[0] * M[0] + M[1] * M[1] + M[2] * M[2]),
							sqrtf(M[4] * M[4] + M[5] * M[5] + M[6] * M[6]),
							sqrtf(M[8] * M[8] + M[9] * M[9] + M[10] * M[10]));
	}

	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setRotationDegrees( const vector3d<T>& rotation )
	{
		return setRotationRadians( rotation * core::DEGTORAD );
	}

	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setInverseRotationDegrees( const vector3d<T>& rotation )
	{
		return setInverseRotationRadians( rotation * core::DEGTORAD );
	}

	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setRotationRadians( const vector3d<T>& rotation )
	{
		const double cr = cos( rotation.X );
		const double sr = sin( rotation.X );
		const double cp = cos( rotation.Y );
		const double sp = sin( rotation.Y );
		const double cy = cos( rotation.Z );
		const double sy = sin( rotation.Z );

		M[0] = (T)( cp*cy );
		M[1] = (T)( cp*sy );
		M[2] = (T)( -sp );

		const double srsp = sr*sp;
		const double crsp = cr*sp;

		M[4] = (T)( srsp*cy-cr*sy );
		M[5] = (T)( srsp*sy+cr*cy );
		M[6] = (T)( sr*cp );

		M[8] = (T)( crsp*cy+sr*sy );
		M[9] = (T)( crsp*sy-sr*cy );
		M[10] = (T)( cr*cp );
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	//! Returns a rotation that is equivalent to that set by setRotationDegrees().
	/** This code was sent in by Chev.  Note that it does not necessarily return
	the *same* Euler angles as those set by setRotationDegrees(), but the rotation will
	be equivalent, i.e. will have the same result when used to rotate a vector or node. */
	template <class T>
	inline core::vector3d<T> CMatrix4<T>::getRotationDegrees() const
	{
		const CMatrix4<T> &mat = *this;
		core::vector3d<T> scale = getScale();
		// we need to check for negative scale on to axes, which would bring up wrong results
		if (scale.Y<0 && scale.Z<0)
		{
			scale.Y =-scale.Y;
			scale.Z =-scale.Z;
		}
		else if (scale.X<0 && scale.Z<0)
		{
			scale.X =-scale.X;
			scale.Z =-scale.Z;
		}
		else if (scale.X<0 && scale.Y<0)
		{
			scale.X =-scale.X;
			scale.Y =-scale.Y;
		}
		const core::vector3d<double> invScale(core::reciprocal(scale.X),core::reciprocal(scale.Y),core::reciprocal(scale.Z));

		double Y = -asin(core::clamp(mat[2]*invScale.X, -1.0, 1.0));
		const double C = cos(Y);
		Y *= RADTODEG64;

		double rotx, roty, X, Z;

		if (!core::iszero(C))
		{
			const double invC = core::reciprocal(C);
			rotx = mat[10] * invC * invScale.Z;
			roty = mat[6] * invC * invScale.Y;
			X = atan2( roty, rotx ) * RADTODEG64;
			rotx = mat[0] * invC * invScale.X;
			roty = mat[1] * invC * invScale.X;
			Z = atan2( roty, rotx ) * RADTODEG64;
		}
		else
		{
			X = 0.0;
			rotx = mat[5] * invScale.Y;
			roty = -mat[4] * invScale.Y;
			Z = atan2( roty, rotx ) * RADTODEG64;
		}

		// fix values that get below zero
		if (X < 0.0) X += 360.0;
		if (Y < 0.0) Y += 360.0;
		if (Z < 0.0) Z += 360.0;

		return vector3d<T>((T)X,(T)Y,(T)Z);
	}


	//! Sets matrix to rotation matrix of inverse angles given as parameters
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setInverseRotationRadians( const vector3d<T>& rotation )
	{
		double cr = cos( rotation.X );
		double sr = sin( rotation.X );
		double cp = cos( rotation.Y );
		double sp = sin( rotation.Y );
		double cy = cos( rotation.Z );
		double sy = sin( rotation.Z );

		M[0] = (T)( cp*cy );
		M[4] = (T)( cp*sy );
		M[8] = (T)( -sp );

		double srsp = sr*sp;
		double crsp = cr*sp;

		M[1] = (T)( srsp*cy-cr*sy );
		M[5] = (T)( srsp*sy+cr*cy );
		M[9] = (T)( sr*cp );

		M[2] = (T)( crsp*cy+sr*sy );
		M[6] = (T)( crsp*sy-sr*cy );
		M[10] = (T)( cr*cp );
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}

	//! Sets matrix to rotation matrix defined by axis and angle, assuming LH rotation
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setRotationAxisRadians( const T& angle, const vector3d<T>& axis )
	{
 		const double c = cos(angle);
		const double s = sin(angle);
		const double t = 1.0 - c;

		const double tx  = t * axis.X;
		const double ty  = t * axis.Y;
		const double tz  = t * axis.Z;

		const double sx  = s * axis.X;
		const double sy  = s * axis.Y;
		const double sz  = s * axis.Z;

		M[0] = (T)(tx * axis.X + c);
		M[1] = (T)(tx * axis.Y + sz);
		M[2] = (T)(tx * axis.Z - sy);

		M[4] = (T)(ty * axis.X - sz);
		M[5] = (T)(ty * axis.Y + c);
		M[6] = (T)(ty * axis.Z + sx);

		M[8]  = (T)(tz * axis.X + sy);
		M[9]  = (T)(tz * axis.Y - sx);
		M[10] = (T)(tz * axis.Z + c);

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	/*!
	*/
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::makeIdentity()
	{
		memset(M, 0, 16*sizeof(T));
		M[0] = M[5] = M[10] = M[15] = (T)1;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=true;
#endif
		return *this;
	}


	/*
		check identity with epsilon
		solve floating range problems..
	*/
	template <class T>
	inline bool CMatrix4<T>::isIdentity() const
	{
#if defined ( USE_MATRIX_TEST )
		if (definitelyIdentityMatrix)
			return true;
#endif
		if (!core::equals( M[12], (T)0 ) || !core::equals( M[13], (T)0 ) || !core::equals( M[14], (T)0 ) || !core::equals( M[15], (T)1 ))
			return false;

		if (!core::equals( M[ 0], (T)1 ) || !core::equals( M[ 1], (T)0 ) || !core::equals( M[ 2], (T)0 ) || !core::equals( M[ 3], (T)0 ))
			return false;

		if (!core::equals( M[ 4], (T)0 ) || !core::equals( M[ 5], (T)1 ) || !core::equals( M[ 6], (T)0 ) || !core::equals( M[ 7], (T)0 ))
			return false;

		if (!core::equals( M[ 8], (T)0 ) || !core::equals( M[ 9], (T)0 ) || !core::equals( M[10], (T)1 ) || !core::equals( M[11], (T)0 ))
			return false;
/*
		if (!core::equals( M[ 0], (T)1 ) ||
			!core::equals( M[ 5], (T)1 ) ||
			!core::equals( M[10], (T)1 ) ||
			!core::equals( M[15], (T)1 ))
			return false;

		for (int32_t i=0; i<4; ++i)
			for (int32_t j=0; j<4; ++j)
				if ((j != i) && (!iszero((*this)(i,j))))
					return false;
*/
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=true;
#endif
		return true;
	}


	/* Check orthogonality of matrix. */
	template <class T>
	inline bool CMatrix4<T>::isOrthogonal() const
	{
		T dp=M[0] * M[4 ] + M[1] * M[5 ] + M[2 ] * M[6 ] + M[3 ] * M[7 ];
		if (!iszero(dp))
			return false;
		dp = M[0] * M[8 ] + M[1] * M[9 ] + M[2 ] * M[10] + M[3 ] * M[11];
		if (!iszero(dp))
			return false;
		dp = M[0] * M[12] + M[1] * M[13] + M[2 ] * M[14] + M[3 ] * M[15];
		if (!iszero(dp))
			return false;
		dp = M[4] * M[8 ] + M[5] * M[9 ] + M[6 ] * M[10] + M[7 ] * M[11];
		if (!iszero(dp))
			return false;
		dp = M[4] * M[12] + M[5] * M[13] + M[6 ] * M[14] + M[7 ] * M[15];
		if (!iszero(dp))
			return false;
		dp = M[8] * M[12] + M[9] * M[13] + M[10] * M[14] + M[11] * M[15];
		return (iszero(dp));
	}


	/*
		doesn't solve floating range problems..
		but takes care on +/- 0 on translation because we are changing it..
		reducing floating point branches
		but it needs the floats in memory..
	*/
	template <class T>
	inline bool CMatrix4<T>::isIdentity_integer_base() const
	{
#if defined ( USE_MATRIX_TEST )
		if (definitelyIdentityMatrix)
			return true;
#endif
		if(IR(M[0])!=F32_VALUE_1)	return false;
		if(IR(M[1])!=0)			return false;
		if(IR(M[2])!=0)			return false;
		if(IR(M[3])!=0)			return false;

		if(IR(M[4])!=0)			return false;
		if(IR(M[5])!=F32_VALUE_1)	return false;
		if(IR(M[6])!=0)			return false;
		if(IR(M[7])!=0)			return false;

		if(IR(M[8])!=0)			return false;
		if(IR(M[9])!=0)			return false;
		if(IR(M[10])!=F32_VALUE_1)	return false;
		if(IR(M[11])!=0)		return false;

		if(IR(M[12])!=0)		return false;
		if(IR(M[13])!=0)		return false;
		if(IR(M[13])!=0)		return false;
		if(IR(M[15])!=F32_VALUE_1)	return false;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=true;
#endif
		return true;
	}


	template <class T>
	inline void CMatrix4<T>::rotateVect( vector3df& vect ) const
	{
		vector3df tmp = vect;
		vect.X = tmp.X*M[0] + tmp.Y*M[4] + tmp.Z*M[8];
		vect.Y = tmp.X*M[1] + tmp.Y*M[5] + tmp.Z*M[9];
		vect.Z = tmp.X*M[2] + tmp.Y*M[6] + tmp.Z*M[10];
	}

	//! An alternate transform vector method, writing into a second vector
	template <class T>
	inline void CMatrix4<T>::rotateVect(core::vector3df& out, const core::vector3df& in) const
	{
		out.X = in.X*M[0] + in.Y*M[4] + in.Z*M[8];
		out.Y = in.X*M[1] + in.Y*M[5] + in.Z*M[9];
		out.Z = in.X*M[2] + in.Y*M[6] + in.Z*M[10];
	}

	//! An alternate transform vector method, writing into an array of 3 floats
	template <class T>
	inline void CMatrix4<T>::rotateVect(T *out, const core::vector3df& in) const
	{
		out[0] = in.X*M[0] + in.Y*M[4] + in.Z*M[8];
		out[1] = in.X*M[1] + in.Y*M[5] + in.Z*M[9];
		out[2] = in.X*M[2] + in.Y*M[6] + in.Z*M[10];
	}

	template <class T>
	inline void CMatrix4<T>::transformVect( vector3df& vect) const
	{
		float vector[3];

		vector[0] = vect.X*M[0] + vect.Y*M[4] + vect.Z*M[8] + M[12];
		vector[1] = vect.X*M[1] + vect.Y*M[5] + vect.Z*M[9] + M[13];
		vector[2] = vect.X*M[2] + vect.Y*M[6] + vect.Z*M[10] + M[14];

		vect.X = vector[0];
		vect.Y = vector[1];
		vect.Z = vector[2];
	}

	template <class T>
	inline void CMatrix4<T>::transformVect( vector3df& out, const vector3df& in) const
	{
		out.X = in.X*M[0] + in.Y*M[4] + in.Z*M[8] + M[12];
		out.Y = in.X*M[1] + in.Y*M[5] + in.Z*M[9] + M[13];
		out.Z = in.X*M[2] + in.Y*M[6] + in.Z*M[10] + M[14];
	}


	template <class T>
	inline void CMatrix4<T>::transformVect(T *out, const core::vector3df &in) const
	{
		out[0] = in.X*M[0] + in.Y*M[4] + in.Z*M[8] + M[12];
		out[1] = in.X*M[1] + in.Y*M[5] + in.Z*M[9] + M[13];
		out[2] = in.X*M[2] + in.Y*M[6] + in.Z*M[10] + M[14];
		out[3] = in.X*M[3] + in.Y*M[7] + in.Z*M[11] + M[15];
	}

	template <class T>
	inline void CMatrix4<T>::transformVec3(T *out, const T * in) const
	{
		out[0] = in[0]*M[0] + in[1]*M[4] + in[2]*M[8] + M[12];
		out[1] = in[0]*M[1] + in[1]*M[5] + in[2]*M[9] + M[13];
		out[2] = in[0]*M[2] + in[1]*M[6] + in[2]*M[10] + M[14];
	}


	//! Transforms a plane by this matrix
	template <class T>
	inline void CMatrix4<T>::transformPlane( core::plane3d<float> &plane) const
	{
		core::plane3df temp;
		transformPlane(plane,temp);
		plane = temp;
	}

	//! Transforms a plane by this matrix
	template <class T>
	inline void CMatrix4<T>::transformPlane( const core::plane3d<float> &in, core::plane3d<float> &out) const
	{
		CMatrix4<T> transposedInverse(*this, EM4CONST_INVERSE);
        out.Normal.X = in.Normal.X*transposedInverse[0] + in.Normal.Y*transposedInverse[1] + in.Normal.Z*transposedInverse[2] + in.D*transposedInverse[3];
        out.Normal.Y = in.Normal.X*transposedInverse[4] + in.Normal.Y*transposedInverse[5] + in.Normal.Z*transposedInverse[6] + in.D*transposedInverse[7];
        out.Normal.Z = in.Normal.X*transposedInverse[8] + in.Normal.Y*transposedInverse[9] + in.Normal.Z*transposedInverse[10] + in.D*transposedInverse[11];
        out.D = in.Normal.X*transposedInverse[12] + in.Normal.Y*transposedInverse[13] + in.Normal.Z*transposedInverse[14] + in.D*transposedInverse[15];
	}


	//! Multiplies this matrix by a 1x4 matrix
	template <class T>
	inline void CMatrix4<T>::multiplyWith1x4Matrix(T* matrix) const
	{
		/*
		0  1  2  3
		4  5  6  7
		8  9  10 11
		12 13 14 15
		*/

		T mat[4];
		mat[0] = matrix[0];
		mat[1] = matrix[1];
		mat[2] = matrix[2];
		mat[3] = matrix[3];

		matrix[0] = M[0]*mat[0] + M[4]*mat[1] + M[8]*mat[2] + M[12]*mat[3];
		matrix[1] = M[1]*mat[0] + M[5]*mat[1] + M[9]*mat[2] + M[13]*mat[3];
		matrix[2] = M[2]*mat[0] + M[6]*mat[1] + M[10]*mat[2] + M[14]*mat[3];
		matrix[3] = M[3]*mat[0] + M[7]*mat[1] + M[11]*mat[2] + M[15]*mat[3];
	}

	template <class T>
	inline void CMatrix4<T>::inverseTranslateVect( vector3df& vect ) const
	{
		vect.X = vect.X-M[12];
		vect.Y = vect.Y-M[13];
		vect.Z = vect.Z-M[14];
	}

	template <class T>
	inline void CMatrix4<T>::translateVect( vector3df& vect ) const
	{
		vect.X = vect.X+M[12];
		vect.Y = vect.Y+M[13];
		vect.Z = vect.Z+M[14];
	}


	template <class T>
	inline bool CMatrix4<T>::getInverse(CMatrix4<T>& out) const
	{
		/// Calculates the inverse of this Matrix
		/// The inverse is calculated using Cramers rule.
		/// If no inverse exists then 'false' is returned.

#if defined ( USE_MATRIX_TEST )
		if ( this->isIdentity() )
		{
			out=*this;
			return true;
		}
#endif
		const CMatrix4<T> &m = *this;

		float d = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2)) -
			(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)) * (m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1)) +
			(m(0, 0) * m(1, 3) - m(0, 3) * m(1, 0)) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1)) +
			(m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * (m(2, 0) * m(3, 3) - m(2, 3) * m(3, 0)) -
			(m(0, 1) * m(1, 3) - m(0, 3) * m(1, 1)) * (m(2, 0) * m(3, 2) - m(2, 2) * m(3, 0)) +
			(m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2)) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0));

		if( core::iszero ( d, FLT_MIN ) )
			return false;

		d = core::reciprocal ( d );

		out(0, 0) = d * (m(1, 1) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2)) +
				m(1, 2) * (m(2, 3) * m(3, 1) - m(2, 1) * m(3, 3)) +
				m(1, 3) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1)));
		out(0, 1) = d * (m(2, 1) * (m(0, 2) * m(3, 3) - m(0, 3) * m(3, 2)) +
				m(2, 2) * (m(0, 3) * m(3, 1) - m(0, 1) * m(3, 3)) +
				m(2, 3) * (m(0, 1) * m(3, 2) - m(0, 2) * m(3, 1)));
		out(0, 2) = d * (m(3, 1) * (m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2)) +
				m(3, 2) * (m(0, 3) * m(1, 1) - m(0, 1) * m(1, 3)) +
				m(3, 3) * (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)));
		out(0, 3) = d * (m(0, 1) * (m(1, 3) * m(2, 2) - m(1, 2) * m(2, 3)) +
				m(0, 2) * (m(1, 1) * m(2, 3) - m(1, 3) * m(2, 1)) +
				m(0, 3) * (m(1, 2) * m(2, 1) - m(1, 1) * m(2, 2)));
		out(1, 0) = d * (m(1, 2) * (m(2, 0) * m(3, 3) - m(2, 3) * m(3, 0)) +
				m(1, 3) * (m(2, 2) * m(3, 0) - m(2, 0) * m(3, 2)) +
				m(1, 0) * (m(2, 3) * m(3, 2) - m(2, 2) * m(3, 3)));
		out(1, 1) = d * (m(2, 2) * (m(0, 0) * m(3, 3) - m(0, 3) * m(3, 0)) +
				m(2, 3) * (m(0, 2) * m(3, 0) - m(0, 0) * m(3, 2)) +
				m(2, 0) * (m(0, 3) * m(3, 2) - m(0, 2) * m(3, 3)));
		out(1, 2) = d * (m(3, 2) * (m(0, 0) * m(1, 3) - m(0, 3) * m(1, 0)) +
				m(3, 3) * (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) +
				m(3, 0) * (m(0, 3) * m(1, 2) - m(0, 2) * m(1, 3)));
		out(1, 3) = d * (m(0, 2) * (m(1, 3) * m(2, 0) - m(1, 0) * m(2, 3)) +
				m(0, 3) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
				m(0, 0) * (m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2)));
		out(2, 0) = d * (m(1, 3) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0)) +
				m(1, 0) * (m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1)) +
				m(1, 1) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)));
		out(2, 1) = d * (m(2, 3) * (m(0, 0) * m(3, 1) - m(0, 1) * m(3, 0)) +
				m(2, 0) * (m(0, 1) * m(3, 3) - m(0, 3) * m(3, 1)) +
				m(2, 1) * (m(0, 3) * m(3, 0) - m(0, 0) * m(3, 3)));
		out(2, 2) = d * (m(3, 3) * (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) +
				m(3, 0) * (m(0, 1) * m(1, 3) - m(0, 3) * m(1, 1)) +
				m(3, 1) * (m(0, 3) * m(1, 0) - m(0, 0) * m(1, 3)));
		out(2, 3) = d * (m(0, 3) * (m(1, 1) * m(2, 0) - m(1, 0) * m(2, 1)) +
				m(0, 0) * (m(1, 3) * m(2, 1) - m(1, 1) * m(2, 3)) +
				m(0, 1) * (m(1, 0) * m(2, 3) - m(1, 3) * m(2, 0)));
		out(3, 0) = d * (m(1, 0) * (m(2, 2) * m(3, 1) - m(2, 1) * m(3, 2)) +
				m(1, 1) * (m(2, 0) * m(3, 2) - m(2, 2) * m(3, 0)) +
				m(1, 2) * (m(2, 1) * m(3, 0) - m(2, 0) * m(3, 1)));
		out(3, 1) = d * (m(2, 0) * (m(0, 2) * m(3, 1) - m(0, 1) * m(3, 2)) +
				m(2, 1) * (m(0, 0) * m(3, 2) - m(0, 2) * m(3, 0)) +
				m(2, 2) * (m(0, 1) * m(3, 0) - m(0, 0) * m(3, 1)));
		out(3, 2) = d * (m(3, 0) * (m(0, 2) * m(1, 1) - m(0, 1) * m(1, 2)) +
				m(3, 1) * (m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)) +
				m(3, 2) * (m(0, 1) * m(1, 0) - m(0, 0) * m(1, 1)));
		out(3, 3) = d * (m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) +
				m(0, 1) * (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) +
				m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)));

#if defined ( USE_MATRIX_TEST )
		out.definitelyIdentityMatrix = definitelyIdentityMatrix;
#endif
		return true;
	}


	//! Inverts a primitive matrix which only contains a translation and a rotation
	//! \param out: where result matrix is written to.
	template <class T>
	inline bool CMatrix4<T>::getInversePrimitive ( CMatrix4<T>& out ) const
	{
		out.M[0 ] = M[0];
		out.M[1 ] = M[4];
		out.M[2 ] = M[8];
		out.M[3 ] = 0;

		out.M[4 ] = M[1];
		out.M[5 ] = M[5];
		out.M[6 ] = M[9];
		out.M[7 ] = 0;

		out.M[8 ] = M[2];
		out.M[9 ] = M[6];
		out.M[10] = M[10];
		out.M[11] = 0;

		out.M[12] = (T)-(M[12]*M[0] + M[13]*M[1] + M[14]*M[2]);
		out.M[13] = (T)-(M[12]*M[4] + M[13]*M[5] + M[14]*M[6]);
		out.M[14] = (T)-(M[12]*M[8] + M[13]*M[9] + M[14]*M[10]);
		out.M[15] = 1;

#if defined ( USE_MATRIX_TEST )
		out.definitelyIdentityMatrix = definitelyIdentityMatrix;
#endif
		return true;
	}

	/*!
	*/
	template <class T>
	inline bool CMatrix4<T>::makeInverse()
	{
#if defined ( USE_MATRIX_TEST )
		if (definitelyIdentityMatrix)
			return true;
#endif
		CMatrix4<T> temp ( EM4CONST_NOTHING );

		if (getInverse(temp))
		{
			*this = temp;
			return true;
		}

		return false;
	}


	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::operator=(const CMatrix4<T> &other)
	{
		if (this==&other)
			return *this;
		memcpy(M, other.M, 16*sizeof(T));
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=other.definitelyIdentityMatrix;
#endif
		return *this;
	}


	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::operator=(const T& scalar)
	{
		for (int32_t i = 0; i < 16; ++i)
			M[i]=scalar;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	template <class T>
	inline bool CMatrix4<T>::operator==(const CMatrix4<T> &other) const
	{
#if defined ( USE_MATRIX_TEST )
		if (definitelyIdentityMatrix && other.definitelyIdentityMatrix)
			return true;
#endif
		for (int32_t i = 0; i < 16; ++i)
			if (M[i] != other.M[i])
				return false;

		return true;
	}


	template <class T>
	inline bool CMatrix4<T>::operator!=(const CMatrix4<T> &other) const
	{
		return !(*this == other);
	}


	// Builds a right-handed perspective projection matrix based on a field of view
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::buildProjectionMatrixPerspectiveFovRH(
			float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
	{
		const double h = reciprocal(tan(fieldOfViewRadians*0.5));
		_IRR_DEBUG_BREAK_IF(aspectRatio==0.f); //divide by zero
		const T w = static_cast<T>(h / aspectRatio);

		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = w;
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = (T)-h;
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (T)(zFar/(zFar-zNear));
		M[11] = 1;

		M[12] = 0;
		M[13] = 0;
		M[14] = (T)(-zNear*zFar/(zFar-zNear));
		M[15] = 0;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a left-handed perspective projection matrix based on a field of view
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::buildProjectionMatrixPerspectiveFovLH(
			float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
	{
		const double h = reciprocal(tan(fieldOfViewRadians*0.5));
		_IRR_DEBUG_BREAK_IF(aspectRatio==0.f); //divide by zero
		const T w = static_cast<T>(h / aspectRatio);

		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = w;
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = (T)h;
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (T)(zFar/(zFar-zNear));
		M[11] = 1;

		M[12] = 0;
		M[13] = 0;
		M[14] = (T)(-zNear*zFar/(zFar-zNear));
		M[15] = 0;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}

	// Builds a right-handed orthogonal projection matrix.
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::buildProjectionMatrixOrthoRH(
			float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
	{
		_IRR_DEBUG_BREAK_IF(widthOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(heightOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = (T)(2.f/widthOfViewVolume);
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = -(T)(2.f/heightOfViewVolume);
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (T)(1/(zFar-zNear));
		M[11] = 0;

		M[12] = 0;
		M[13] = 0;
		M[14] = (T)(-zNear/(zFar-zNear));
		M[15] = 1;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}

	// Builds a left-handed orthogonal projection matrix.
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::buildProjectionMatrixOrthoLH(
			float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
	{
		_IRR_DEBUG_BREAK_IF(widthOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(heightOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = (T)(2/widthOfViewVolume);
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = (T)(2/heightOfViewVolume);
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (T)(1/(zFar-zNear));
		M[11] = 0;

		M[12] = 0;
		M[13] = 0;
		M[14] = (T)(-zNear/(zFar-zNear));
		M[15] = 1;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a left-handed look-at matrix.
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::buildCameraLookAtMatrixLH(
				const vector3df& position,
				const vector3df& target,
				const vector3df& upVector)
	{
		vector3df zaxis = target - position;
		zaxis.normalize();

		vector3df xaxis = upVector.crossProduct(zaxis);
		xaxis.normalize();

		vector3df yaxis = zaxis.crossProduct(xaxis);

		M[0] = (T)xaxis.X;
		M[1] = (T)yaxis.X;
		M[2] = (T)zaxis.X;
		M[3] = 0;

		M[4] = (T)xaxis.Y;
		M[5] = (T)yaxis.Y;
		M[6] = (T)zaxis.Y;
		M[7] = 0;

		M[8] = (T)xaxis.Z;
		M[9] = (T)yaxis.Z;
		M[10] = (T)zaxis.Z;
		M[11] = 0;

		M[12] = (T)-xaxis.dotProduct(position);
		M[13] = (T)-yaxis.dotProduct(position);
		M[14] = (T)-zaxis.dotProduct(position);
		M[15] = 1;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a right-handed look-at matrix.
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::buildCameraLookAtMatrixRH(
				const vector3df& position,
				const vector3df& target,
				const vector3df& upVector)
	{
		vector3df zaxis = position - target;
		zaxis.normalize();

		vector3df xaxis = upVector.crossProduct(zaxis);
		xaxis.normalize();

		vector3df yaxis = zaxis.crossProduct(xaxis);

		M[0] = (T)xaxis.X;
		M[1] = (T)yaxis.X;
		M[2] = (T)zaxis.X;
		M[3] = 0;

		M[4] = (T)xaxis.Y;
		M[5] = (T)yaxis.Y;
		M[6] = (T)zaxis.Y;
		M[7] = 0;

		M[8] = (T)xaxis.Z;
		M[9] = (T)yaxis.Z;
		M[10] = (T)zaxis.Z;
		M[11] = 0;

		M[12] = (T)-xaxis.dotProduct(position);
		M[13] = (T)-yaxis.dotProduct(position);
		M[14] = (T)-zaxis.dotProduct(position);
		M[15] = 1;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// creates a new matrix as interpolated matrix from this and the passed one.
	template <class T>
	inline CMatrix4<T> CMatrix4<T>::interpolate(const core::CMatrix4<T>& b, float time) const
	{
		CMatrix4<T> mat ( EM4CONST_NOTHING );

		for (uint32_t i=0; i < 16; i += 4)
		{
			mat.M[i+0] = (T)(M[i+0] + ( b.M[i+0] - M[i+0] ) * time);
			mat.M[i+1] = (T)(M[i+1] + ( b.M[i+1] - M[i+1] ) * time);
			mat.M[i+2] = (T)(M[i+2] + ( b.M[i+2] - M[i+2] ) * time);
			mat.M[i+3] = (T)(M[i+3] + ( b.M[i+3] - M[i+3] ) * time);
		}
		return mat;
	}


	// returns transposed matrix
	template <class T>
	inline CMatrix4<T> CMatrix4<T>::getTransposed() const
	{
		CMatrix4<T> t ( EM4CONST_NOTHING );
		getTransposed ( t );
		return t;
	}


	// returns transposed matrix
	template <class T>
	inline void CMatrix4<T>::getTransposed( CMatrix4<T>& o ) const
	{
		o[ 0] = M[ 0];
		o[ 1] = M[ 4];
		o[ 2] = M[ 8];
		o[ 3] = M[12];

		o[ 4] = M[ 1];
		o[ 5] = M[ 5];
		o[ 6] = M[ 9];
		o[ 7] = M[13];

		o[ 8] = M[ 2];
		o[ 9] = M[ 6];
		o[10] = M[10];
		o[11] = M[14];

		o[12] = M[ 3];
		o[13] = M[ 7];
		o[14] = M[11];
		o[15] = M[15];
#if defined ( USE_MATRIX_TEST )
		o.definitelyIdentityMatrix=definitelyIdentityMatrix;
#endif
	}

	//! Builds a combined matrix which translate to a center before rotation and translate afterwards
	template <class T>
	inline void CMatrix4<T>::setRotationCenter(const core::vector3df& center, const core::vector3df& translation)
	{
		M[12] = -M[0]*center.X - M[4]*center.Y - M[8]*center.Z + (center.X - translation.X );
		M[13] = -M[1]*center.X - M[5]*center.Y - M[9]*center.Z + (center.Y - translation.Y );
		M[14] = -M[2]*center.X - M[6]*center.Y - M[10]*center.Z + (center.Z - translation.Z );
		M[15] = (T) 1.0;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
	}


	// rotate about z axis, center ( 0.5, 0.5 )
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setTextureRotationCenter( float rotateRad )
	{
		const float c = cosf(rotateRad);
		const float s = sinf(rotateRad);
		M[0] = (T)c;
		M[1] = (T)s;

		M[4] = (T)-s;
		M[5] = (T)c;

		M[8] = (T)(0.5f * ( s - c) + 0.5f);
		M[9] = (T)(-0.5f * ( s + c) + 0.5f);

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (rotateRad==0.0f);
#endif
		return *this;
	}


	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setTextureTranslate ( float x, float y )
	{
		M[8] = (T)x;
		M[9] = (T)y;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (x==0.0f) && (y==0.0f);
#endif
		return *this;
	}


	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setTextureTranslateTransposed ( float x, float y )
	{
		M[2] = (T)x;
		M[6] = (T)y;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (x==0.0f) && (y==0.0f) ;
#endif
		return *this;
	}

	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setTextureScale ( float sx, float sy )
	{
		M[0] = (T)sx;
		M[5] = (T)sy;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (sx==1.0f) && (sy==1.0f);
#endif
		return *this;
	}


	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setTextureScaleCenter( float sx, float sy )
	{
		M[0] = (T)sx;
		M[5] = (T)sy;
		M[8] = (T)(0.5f - 0.5f * sx);
		M[9] = (T)(0.5f - 0.5f * sy);

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (sx==1.0f) && (sy==1.0f);
#endif
		return *this;
	}


	// sets all matrix data members at once
	template <class T>
	inline CMatrix4<T>& CMatrix4<T>::setM(const T* data)
	{
		memcpy(M,data, 16*sizeof(T));

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// sets if the matrix is definitely identity matrix
	template <class T>
	inline void CMatrix4<T>::setDefinitelyIdentityMatrix( bool isDefinitelyIdentityMatrix)
	{
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = isDefinitelyIdentityMatrix;
#endif
	}


	// gets if the matrix is definitely identity matrix
	template <class T>
	inline bool CMatrix4<T>::getDefinitelyIdentityMatrix() const
	{
#if defined ( USE_MATRIX_TEST )
		return definitelyIdentityMatrix;
#else
		return false;
#endif
	}


	//! Compare two matrices using the equal method
	template <class T>
	inline bool CMatrix4<T>::equals(const core::CMatrix4<T>& other, const T tolerance) const
	{
#if defined ( USE_MATRIX_TEST )
		if (definitelyIdentityMatrix && other.definitelyIdentityMatrix)
			return true;
#endif
		for (int32_t i = 0; i < 16; ++i)
			if (!core::equals(M[i],other.M[i], tolerance))
				return false;

		return true;
	}


	// Multiply by scalar.
	template <class T>
	inline CMatrix4<T> operator*(const T scalar, const CMatrix4<T>& mat)
	{
		return mat*scalar;
	}


	//! Typedef for float matrix
	typedef CMatrix4<float> matrix4;

} // end namespace core
} // end namespace irr

#endif

