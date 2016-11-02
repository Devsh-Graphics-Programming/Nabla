
// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MATRIX_H_INCLUDED__
#define __IRR_MATRIX_H_INCLUDED__

#define __IRR_COMPILE_WITH_X86_SIMD_
#ifdef __IRR_COMPILE_WITH_X86_SIMD_

#include "matrix4.h"
#include "vectorSIMD.h"


namespace irr
{
namespace core
{

	//! 4x4 matrix. Mostly used as transformation matrix for 3d calculations.
	/** Translations in the 4th column, this is laid out in memory in the completely opposite way to irrlicht matrix4. */
	class matrixSIMD4
	{
		public:

			//! Default constructor
			/** \param constructor Choose the initialization style */
			matrixSIMD4( matrix4::eConstructor constructor = matrix4::EM4CONST_IDENTITY );
			//! Copy constructor
			/** \param other Other matrix to copy from
			\param constructor Choose the initialization style */
			matrixSIMD4(const matrixSIMD4& other, matrix4::eConstructor constructor = matrix4::EM4CONST_COPY);
			//! init from 4 row vectors
			inline matrixSIMD4(const vectorSIMDf& row0,const vectorSIMDf& row1,const vectorSIMDf& row2,const vectorSIMDf& row3)
			{
			    rows[0] = row0;
			    rows[1] = row1;
			    rows[2] = row2;
			    rows[3] = row3;
			}
			//! init from 16 floats
			inline matrixSIMD4( const float& x0,const float& y0,const float& z0,const float& w0,
                                const float& x1,const float& y1,const float& z1,const float& w1,
                                const float& x2,const float& y2,const float& z2,const float& w2,
                                const float& x3,const float& y3,const float& z3,const float& w3)
            {
                rows[0] = _mm_set_ps(w0,z0,y0,x0);
                rows[1] = _mm_set_ps(w1,z1,y1,x1);
                rows[2] = _mm_set_ps(w2,z2,y2,x2);
                rows[3] = _mm_set_ps(w3,z3,y3,x3);
            }
            //! init from 1 float
			explicit matrixSIMD4( const float& scalar)
            {
                rows[0] = _mm_set1_ps(scalar);
                rows[1] = _mm_set1_ps(scalar);
                rows[2] = _mm_set1_ps(scalar);
                rows[3] = _mm_set1_ps(scalar);
            }
            //! init from 1 float
			explicit matrixSIMD4( const matrix4& retardedIrrlichtMatrix)
            {
                __m128 xmm0 = _mm_loadu_ps(retardedIrrlichtMatrix.pointer);
                __m128 xmm1 = _mm_loadu_ps(retardedIrrlichtMatrix.pointer+4);
                __m128 xmm2 = _mm_loadu_ps(retardedIrrlichtMatrix.pointer+8);
                __m128 xmm3 = _mm_loadu_ps(retardedIrrlichtMatrix.pointer+12);

                _MM_TRANSPOSE4_PS(xmm0,xmm1,xmm2,xmm3);

                rows[0] = xmm0;
                rows[1] = xmm1;
                rows[2] = xmm2;
                rows[3] = xmm3;
            }

            inline matrix4 getAsRetardedIrrlichtMatrix()
            {
                __m128 xmm0 = rows[0].getAsRegister();
                __m128 xmm1 = rows[1].getAsRegister();
                __m128 xmm2 = rows[2].getAsRegister();
                __m128 xmm3 = rows[3].getAsRegister();
                _MM_TRANSPOSE4_PS(xmm0,xmm1,xmm2,xmm3)

#ifdef _IRR_WINDOWS_
                __declspec(align(16)) matrix4 outRIMatrix;
#else
                matrix4 outRIMatrix __attribute__ ((__aligned__(16)));
#endif
                _mm_store_ps(outRIMatrix.pointer,xmm0);
                _mm_store_ps(outRIMatrix.pointer+1,xmm1);
                _mm_store_ps(outRIMatrix.pointer+2,xmm2);
                _mm_store_ps(outRIMatrix.pointer+3,xmm3);

                return outRIMatrix;
            }


			//! Simple operator for directly accessing every element of the matrix.
			inline float& operator()(const s32 &row, const s32 &col)
			{
#if defined ( USE_MATRIX_TEST )
				definitelyIdentityMatrix=false;
#endif
				return rows[row].pointer[col];
			}

			//! Simple operator for directly accessing every element of the matrix.
			inline const float& operator()(const s32 &row, const s32 &col) const { return rows[row].pointer[col]; }

			//! Simple operator for linearly accessing every element of the matrix.
			inline float& operator[](u32 index)
			{
#if defined ( USE_MATRIX_TEST )
				definitelyIdentityMatrix=false;
#endif
				return ((float*)rows[0].pointer)[index];
			}

			//! Simple operator for linearly accessing every element of the matrix.
			inline const float& operator[](u32 index) const { return ((float*)rows[0].pointer)[index]; }

			//! Sets this matrix equal to the other matrix.
			matrixSIMD4& operator=(const matrixSIMD4 &other);

			//! Sets all elements of this matrix to the value.
			matrixSIMD4& operator=(const float& scalar);

			//! Returns pointer to internal array
			inline const float* pointer() const { return rows[0].pointer; }
			inline float* pointer()
			{
#if defined ( USE_MATRIX_TEST )
				definitelyIdentityMatrix=false;
#endif
				return rows[0].pointer;
			}

			//! Returns true if other matrix is equal to this matrix.
			inline bool operator==(const matrixSIMD4 &other) const;

			//! Returns true if other matrix is not equal to this matrix.
			inline bool operator!=(const matrixSIMD4 &other) const;

			//! Add another matrix.
			matrixSIMD4 operator+(const matrixSIMD4& other) const;

			//! Add another matrix.
			matrixSIMD4& operator+=(const matrixSIMD4& other);

			//! Subtract another matrix.
			matrixSIMD4 operator-(const matrixSIMD4& other) const;

			//! Subtract another matrix.
			matrixSIMD4& operator-=(const matrixSIMD4& other);

			//! set this matrix to the product of two matrices
			/** Calculate b*a */
			inline matrixSIMD4& setbyproduct(const matrixSIMD4& other_a,const matrixSIMD4& other_b );

			//! Set this matrix to the product of two matrices
			/** Calculate b*a, no optimization used,
			use it if you know you never have a identity matrix */
			matrixSIMD4& setbyproduct_nocheck(const matrixSIMD4& other_a,const matrixSIMD4& other_b );

			//! Multiply by another matrix.
			/** Calculate other*this */
			matrixSIMD4 operator*(const matrixSIMD4& other) const;

			//! Multiply by another matrix.
			/** Calculate and return other*this */
			matrixSIMD4& operator*=(const matrixSIMD4& other);

			//! Multiply by scalar.
			matrixSIMD4 operator*(const float& scalar) const;

			//! Multiply by scalar.
			matrixSIMD4& operator*=(const float& scalar);

			//! Set matrix to identity.
			inline matrixSIMD4& makeIdentity()
            {
                rows[0] = _mm_set_ps(0,0,0,1);
                rows[1] = _mm_set_ps(0,0,1,0);
                rows[2] = _mm_set_ps(0,1,0,0);
                rows[3] = _mm_set_ps(1,0,0,0);
#if defined ( USE_MATRIX_TEST )
                definitelyIdentityMatrix=true;
#endif
                return *this;
            }


			//! Returns true if the matrix is the identity matrix
			bool isIdentity() const;

			//! Returns true if the matrix is orthogonal
			inline bool isOrthogonal() const;
/*
			//! Returns true if the matrix is the identity matrix
			bool isIdentity_integer_base () const;
*/
			//! Set the translation of the current matrix. Will erase any previous values.
			matrixSIMD4& setTranslation( const vectorSIMDf& translation );

			//! Gets the current translation
			vectorSIMDf getTranslation() const;

			//! Set the inverse translation of the current matrix. Will erase any previous values.
			matrixSIMD4& setInverseTranslation( const vectorSIMDf& translation );
/*
			//! Make a rotation matrix from Euler angles. The 4th row and column are unmodified.
			inline matrixSIMD4& setRotationRadians( const vectorSIMDf& rotation );

			//! Make a rotation matrix from Euler angles. The 4th row and column are unmodified.
			matrixSIMD4& setRotationDegrees( const vectorSIMDf& rotation );

			//! Returns the rotation, as set by setRotation().
			/** This code was orginally written by by Chev. *
			core::vectorSIMDf getRotationDegrees() const;

			//! Make an inverted rotation matrix from Euler angles.
			/** The 4th row and column are unmodified. *
			inline matrixSIMD4& setInverseRotationRadians( const vectorSIMDf& rotation );

			//! Make an inverted rotation matrix from Euler angles.
			/** The 4th row and column are unmodified. *
			inline matrixSIMD4& setInverseRotationDegrees( const vectorSIMDf& rotation );

			//! Make a rotation matrix from angle and axis, assuming left handed rotation.
			/** The 4th row and column are unmodified. *
			inline matrixSIMD4& setRotationAxisRadians(const float& angle, const vectorSIMDf& axis);
*/
			//! Set Scale
			matrixSIMD4& setScale( const vectorSIMDf& scale );

			//! Set Scale
			matrixSIMD4& setScale( const float scale ) { return setScale(_mm_set1_ps(scale)); }

			//! Get Scale
			core::vectorSIMDf getScale() const;

			//! Translate a vector by the inverse of the translation part of this matrix.
			void inverseTranslateVect( vector3df& vect ) const;
/*

			//! Rotate a vector by the rotation part of this matrix.
			void rotateVect( vector3df& vect ) const;

			//! An alternate transform vector method, writing into a second vector
			void rotateVect(core::vector3df& out, const core::vector3df& in) const;

			//! An alternate transform vector method, writing into an array of 3 floats
			void rotateVect(float *out,const core::vector3df &in) const;
*/
			//! Transforms the vector by this matrix
			void transformVect( vector3df& vect) const;

			//! Transforms input vector by this matrix and stores result in output vector
			void transformVect( vector3df& out, const vector3df& in ) const;

			//! An alternate transform vector method, writing into an array of 4 floats
			void transformVect(float *out,const core::vector3df &in) const;

			//! An alternate transform vector method, reading from and writing to an array of 3 floats
			void transformVec3(float *out, const float * in) const;

			//! Translate a vector by the translation part of this matrix.
			void translateVect( vector3df& vect ) const;
/*
			//! Transforms a plane by this matrix
			void transformPlane( core::plane3d<f32> &plane) const;

			//! Transforms a plane by this matrix
			void transformPlane( const core::plane3d<f32> &in, core::plane3d<f32> &out) const;

			//! Transforms a axis aligned bounding box
			/** The result box of this operation may not be accurate at all. For
			correct results, use transformBoxEx() *
			void transformBox(core::aabbox3d<f32>& box) const;

			//! Transforms a axis aligned bounding box
			/** The result box of this operation should by accurate, but this operation
			is slower than transformBox(). *
			void transformBoxEx(core::aabbox3d<f32>& box) const;

			//! Multiplies this matrix by a 1x4 matrix
			void multiplyWith1x4Matrix(T* matrix) const;

			//! Calculates inverse of matrix. Slow.
			/** \return Returns false if there is no inverse matrix.*
			bool makeInverse();


			//! Inverts a primitive matrix which only contains a translation and a rotation
			/** \param out: where result matrix is written to. *
			bool getInversePrimitive ( matrixSIMD4& out ) const;

			//! Gets the inversed matrix of this one
			/** \param out: where result matrix is written to.
			\return Returns false if there is no inverse matrix. *
			bool getInverse(matrixSIMD4& out) const;

			//! Builds a right-handed perspective projection matrix based on a field of view
			matrixSIMD4& buildProjectionMatrixPerspectiveFovRH(f32 fieldOfViewRadians, f32 aspectRatio, f32 zNear, f32 zFar);

			//! Builds a left-handed perspective projection matrix based on a field of view
			matrixSIMD4& buildProjectionMatrixPerspectiveFovLH(f32 fieldOfViewRadians, f32 aspectRatio, f32 zNear, f32 zFar);

			//! Builds a left-handed perspective projection matrix based on a field of view, with far plane at infinity
			matrixSIMD4& buildProjectionMatrixPerspectiveFovInfinityLH(f32 fieldOfViewRadians, f32 aspectRatio, f32 zNear, f32 epsilon=0);

			//! Builds a right-handed perspective projection matrix.
			matrixSIMD4& buildProjectionMatrixPerspectiveRH(f32 widthOfViewVolume, f32 heightOfViewVolume, f32 zNear, f32 zFar);

			//! Builds a left-handed perspective projection matrix.
			matrixSIMD4& buildProjectionMatrixPerspectiveLH(f32 widthOfViewVolume, f32 heightOfViewVolume, f32 zNear, f32 zFar);

			//! Builds a left-handed orthogonal projection matrix.
			matrixSIMD4& buildProjectionMatrixOrthoLH(f32 widthOfViewVolume, f32 heightOfViewVolume, f32 zNear, f32 zFar);

			//! Builds a right-handed orthogonal projection matrix.
			matrixSIMD4& buildProjectionMatrixOrthoRH(f32 widthOfViewVolume, f32 heightOfViewVolume, f32 zNear, f32 zFar);

			//! Builds a left-handed look-at matrix.
			matrixSIMD4& buildCameraLookAtMatrixLH(
					const vector3df& position,
					const vector3df& target,
					const vector3df& upVector);

			//! Builds a right-handed look-at matrix.
			matrixSIMD4& buildCameraLookAtMatrixRH(
					const vector3df& position,
					const vector3df& target,
					const vector3df& upVector);

			//! Builds a matrix that flattens geometry into a plane.
			/** \param light: light source
			\param plane: plane into which the geometry if flattened into
			\param point: value between 0 and 1, describing the light source.
			If this is 1, it is a point light, if it is 0, it is a directional light. *
			matrixSIMD4& buildShadowMatrix(const core::vector3df& light, core::plane3df plane, f32 point=1.0f);

			//! Builds a matrix which transforms a normalized Device Coordinate to Device Coordinates.
			/** Used to scale <-1,-1><1,1> to viewport, for example from <-1,-1> <1,1> to the viewport <0,0><0,640> *
			matrixSIMD4& buildNDCToDCMatrix( const core::rect<s32>& area, f32 zScale);
*/
			//! Creates a new matrix as interpolated matrix from two other ones.
			/** \param b: other matrix to interpolate with
			\param time: Must be a value between 0 and 1. */
			matrixSIMD4 interpolate(const core::matrixSIMD4& b, float factor) const;

			//! Gets transposed matrix
			matrixSIMD4 getTransposed() const;

			//! Gets transposed matrix
			inline void getTransposed( matrixSIMD4& dest ) const;

			//! Builds a matrix that rotates from one vector to another
			/** \param from: vector to rotate from
			\param to: vector to rotate to
			 *
			matrixSIMD4& buildRotateFromTo(const core::vector3df& from, const core::vector3df& to);

			//! Builds a combined matrix which translates to a center before rotation and translates from origin afterwards
			/** \param center Position to rotate around
			\param translate Translation applied after the rotation
			 *
			void setRotationCenter(const core::vector3df& center, const core::vector3df& translate);

			//! Builds a matrix which rotates a source vector to a look vector over an arbitrary axis
			/** \param camPos: viewer position in world coo
			\param center: object position in world-coo and rotation pivot
			\param translation: object final translation from center
			\param axis: axis to rotate about
			\param from: source vector to rotate from
			 *
			void buildAxisAlignedBillboard(const core::vector3df& camPos,
						const core::vector3df& center,
						const core::vector3df& translation,
						const core::vector3df& axis,
						const core::vector3df& from);

			/*
				construct 2D Texture transformations
				rotate about center, scale, and transform.
			*
			//! Set to a texture transformation matrix with the given parameters.
			matrixSIMD4& buildTextureTransform( f32 rotateRad,
					const core::vector2df &rotatecenter,
					const core::vector2df &translate,
					const core::vector2df &scale);

			//! Set texture transformation rotation
			/** Rotate about z axis, recenter at (0.5,0.5).
			Doesn't clear other elements than those affected
			\param radAngle Angle in radians
			\return Altered matrix *
			matrixSIMD4& setTextureRotationCenter( f32 radAngle );

			//! Set texture transformation translation
			/** Doesn't clear other elements than those affected.
			\param x Offset on x axis
			\param y Offset on y axis
			\return Altered matrix *
			matrixSIMD4& setTextureTranslate( f32 x, f32 y );

			//! Set texture transformation translation, using a transposed representation
			/** Doesn't clear other elements than those affected.
			\param x Offset on x axis
			\param y Offset on y axis
			\return Altered matrix *
			matrixSIMD4& setTextureTranslateTransposed( f32 x, f32 y );

			//! Set texture transformation scale
			/** Doesn't clear other elements than those affected.
			\param sx Scale factor on x axis
			\param sy Scale factor on y axis
			\return Altered matrix. *
			matrixSIMD4& setTextureScale( f32 sx, f32 sy );

			//! Set texture transformation scale, and recenter at (0.5,0.5)
			/** Doesn't clear other elements than those affected.
			\param sx Scale factor on x axis
			\param sy Scale factor on y axis
			\return Altered matrix. *
			matrixSIMD4& setTextureScaleCenter( f32 sx, f32 sy );

			//! Sets all matrix data members at once
			matrixSIMD4& setM(const float* data);

			//! Sets if the matrix is definitely identity matrix
			void setDefinitelyIdentityMatrix( bool isDefinitelyIdentityMatrix);

			//! Gets if the matrix is definitely identity matrix
			bool getDefinitelyIdentityMatrix() const;

			//! Compare two matrices using the equal method
			bool equals(const core::matrixSIMD4& other, const float tolerance=ROUNDING_ERROR_f32) const;*/

		private:
			//! Matrix data, stored in row-major order
			vectorSIMDf rows[4];
#if defined ( USE_MATRIX_TEST )
			//! Flag is this matrix is identity matrix
			mutable u32 definitelyIdentityMatrix;
#endif
#if defined ( USE_MATRIX_TEST_DEBUG )
			u32 id;
			mutable u32 calls;
#endif

	};

	// Default constructor
	inline matrixSIMD4::matrixSIMD4( eConstructor constructor )
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
	inline matrixSIMD4::matrixSIMD4( const matrixSIMD4& other, eConstructor constructor)
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
					*this = 0.f;
				break;
			case EM4CONST_INVERSE_TRANSPOSED:
				if (!other.getInverseTransposed(*this))
					*this = 0.f;
				else
					*this = getTransposed();
				break;
		}
	}

	//! Add another matrix.
	inline matrixSIMD4 matrixSIMD4::operator+(const matrixSIMD4& other) const
	{
		matrixSIMD4 temp ( EM4CONST_NOTHING );

		temp.rows[0] = rows[0]+other.rows[0];
		temp.rows[1] = rows[1]+other.rows[1];
		temp.rows[2] = rows[2]+other.rows[2];
		temp.rows[3] = rows[3]+other.rows[3];

		return temp;
	}

	//! Add another matrix.
	inline matrixSIMD4& matrixSIMD4::operator+=(const matrixSIMD4& other)
	{
		rows[0] += other.rows[0];
		rows[1] += other.rows[1];
		rows[2] += other.rows[2];
		rows[3] += other.rows[3];

		return *this;
	}

	//! Subtract another matrix.
	inline matrixSIMD4 matrixSIMD4::operator-(const matrixSIMD4& other) const
	{
		matrixSIMD4 temp ( EM4CONST_NOTHING );

		temp.rows[0] = rows[0]-other.rows[0];
		temp.rows[1] = rows[1]-other.rows[1];
		temp.rows[2] = rows[2]-other.rows[2];
		temp.rows[3] = rows[3]-other.rows[3];

		return temp;
	}

	//! Subtract another matrix.
	inline matrixSIMD4& matrixSIMD4::operator-=(const matrixSIMD4& other)
	{
		rows[0] += other.rows[0];
		rows[1] += other.rows[1];
		rows[2] += other.rows[2];
		rows[3] += other.rows[3];

		return *this;
	}

	//! Multiply by scalar.
	inline matrixSIMD4 matrixSIMD4::operator*(const float& scalar) const
	{
		matrixSIMD4 temp ( EM4CONST_NOTHING );

		temp.rows[0] = rows[0]*scalar;
		temp.rows[1] = rows[1]*scalar;
		temp.rows[2] = rows[2]*scalar;
		temp.rows[3] = rows[3]*scalar;
		return temp;
	}

	//! Multiply by scalar.
	inline matrixSIMD4& matrixSIMD4::operator*=(const float& scalar)
	{
		rows[0] *= scalar;
		rows[1] *= scalar;
		rows[2] *= scalar;
		rows[3] *= scalar;

		return *this;
	}

	//! Multiply by another matrix.
	inline matrixSIMD4& matrixSIMD4::operator*=(const matrixSIMD4& other)
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
				matrixSIMD4 temp ( *this );
				return setbyproduct_nocheck( temp, other );
			}
		}
		return *this;
#else
		matrixSIMD4 temp ( *this );
		return setbyproduct_nocheck( temp, other );
#endif
	}

	//! multiply by another matrix
	// set this matrix to the product of two other matrices
	// goal is to reduce stack use and copy
	inline matrixSIMD4& matrixSIMD4::setbyproduct_nocheck(const matrixSIMD4& other_a,const matrixSIMD4& other_b ) //A*B
	{
	    // xmm4-7 will now become columuns of B
	    __m128 xmm4 = other_b.rows[0].getAsRegister();
	    __m128 xmm5 = other_b.rows[1].getAsRegister();
	    __m128 xmm6 = other_b.rows[2].getAsRegister();
	    __m128 xmm7 = other_b.rows[3].getAsRegister();
	    _MM_TRANSPOSE4_PS(xmm4,xmm5,xmm6,xmm7)


	    __m128 xmm0 = other_a.rows[0].getAsRegister();
        __m128 xmm1 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm4),_mm_mul_ps(xmm0,xmm5)); //(x_l,x_u,y_l,y_u)
        __m128 xmm2 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm6),_mm_mul_ps(xmm0,xmm7)); //(z_l,z_u,w_l,w_u)
        rows[0] = _mm_hadd_ps(xmm1,xmm2); //(x,y,z,w)

	    xmm0 = other_a.rows[1].getAsRegister();
        xmm1 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm4),_mm_mul_ps(xmm0,xmm5)); //(x_l,x_u,y_l,y_u)
        xmm2 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm6),_mm_mul_ps(xmm0,xmm7)); //(z_l,z_u,w_l,w_u)
        rows[1] = _mm_hadd_ps(xmm1,xmm2); //(x,y,z,w)

	    xmm0 = other_a.rows[2].getAsRegister();
        xmm1 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm4),_mm_mul_ps(xmm0,xmm5)); //(x_l,x_u,y_l,y_u)
        xmm2 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm6),_mm_mul_ps(xmm0,xmm7)); //(z_l,z_u,w_l,w_u)
        rows[2] = _mm_hadd_ps(xmm1,xmm2); //(x,y,z,w)

	    xmm0 = other_a.rows[3].getAsRegister();
        xmm1 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm4),_mm_mul_ps(xmm0,xmm5)); //(x_l,x_u,y_l,y_u)
        xmm2 = _mm_hadd_ps(_mm_mul_ps(xmm0,xmm6),_mm_mul_ps(xmm0,xmm7)); //(z_l,z_u,w_l,w_u)
        rows[3] = _mm_hadd_ps(xmm1,xmm2); //(x,y,z,w)

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	//! multiply by another matrix
	// set this matrix to the product of two other matrices
	// goal is to reduce stack use and copy
	inline matrixSIMD4& matrixSIMD4::setbyproduct(const matrixSIMD4& other_a, const matrixSIMD4& other_b )
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
	inline matrixSIMD4 matrixSIMD4::operator*(const matrixSIMD4& m2) const
	{
#if defined ( USE_MATRIX_TEST )
		// Testing purpose..
		if ( this->isIdentity() )
			return m2;
		if ( m2.isIdentity() )
			return *this;

		definitelyIdentityMatrix=false;
#endif


		matrixSIMD4 m3 ( EM4CONST_NOTHING );
		return m3.setbyproduct_nocheck(*this,m2);
	}



	inline vectorSIMDf matrixSIMD4::getTranslation() const
	{
	    __m128 xmm0 = _mm_unpackhi_ps(rows[0].getAsRegister(),rows[1].getAsRegister()); // (0z,1z,0w,1w)
	    __m128 xmm1 = _mm_unpackhi_ps(rows[2].getAsRegister(),rows[3].getAsRegister()); // (2z,3z,2w,3w)
	    __m128 xmm2 = _mm_movehl_ps(xmm1,xmm0);// (0w,1w,2w,3w)

	    return xmm2;
	}
	inline vectorSIMDf matrixSIMD4::getTranslation3D() const
	{
	    __m128 xmm0 = _mm_unpackhi_ps(rows[0].getAsRegister(),rows[1].getAsRegister()); // (0z,1z,0w,1w)
	    __m128 xmm1 = _mm_unpackhi_ps(rows[2].getAsRegister(),_mm_setzero_ps()); // (2z,0,2w,0)
	    __m128 xmm2 = _mm_movehl_ps(xmm1,xmm0);// (0w,1w,2w,0)

	    return xmm2;
	}


	inline matrixSIMD4& matrixSIMD4::setTranslation( const vectorSIMDf& translation )
	{
	    rows[0].W = translation.X;
	    rows[1].W = translation.Y;
	    rows[2].W = translation.Z;
	    rows[3].W = translation.W;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}
	inline matrixSIMD4& matrixSIMD4::setTranslation3D( const vectorSIMDf& translation )
	{
	    rows[0].W = translation.X;
	    rows[1].W = translation.Y;
	    rows[2].W = translation.Z;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	inline matrixSIMD4& matrixSIMD4::setInverseTranslation( const vectorSIMDf& translation )
	{
		return setTranslation(-translation);
	}
	inline matrixSIMD4& matrixSIMD4::setInverseTranslation3D( const vectorSIMDf& translation )
	{
		return setTranslation3D(-translation);
	}

	inline matrixSIMD4& matrixSIMD4::setScale( const vectorSIMDf& scale )
	{
	    //_m128i xmm0 = _mm_castps_si128(_mm_mul_ps(scale.getAsRegister(),_mm_rsqrt_ps(getScaleSQ().getAsRegister())));
	    _m128i xmm0 = _mm_castps_si128(scale.getAsRegister());

	    _mm_maskmoveu_si128(xmm0,_mm_set_epi32(0,0,0,-1),(char*)rows);
	    _mm_maskmoveu_si128(xmm0,_mm_set_epi32(0,0,-1,0),(char*)(rows+1));
	    _mm_maskmoveu_si128(xmm0,_mm_set_epi32(0,-1,0,0),(char*)(rows+2));
	    _mm_maskmoveu_si128(xmm0,_mm_set_epi32(-1,0,0,0),(char*)(rows+3));
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}

	inline matrixSIMD4& matrixSIMD4::setScale3D const vectorSIMDf& scale )
	{
	    //_m128i xmm0 = _mm_castps_si128(_mm_mul_ps(scale.getAsRegister(),_mm_rsqrt_ps(getScaleSQ().getAsRegister())));
	    _m128i xmm0 = _mm_castps_si128(scale.getAsRegister());

	    _mm_maskmoveu_si128(xmm0,_mm_set_epi32(0,0,0,-1),(char*)rows);
	    _mm_maskmoveu_si128(xmm0,_mm_set_epi32(0,0,-1,0),(char*)(rows+1));
	    _mm_maskmoveu_si128(xmm0,_mm_set_epi32(0,-1,0,0),(char*)(rows+2));
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
	inline vectorSIMDf matrixSIMD4::getScaleSQ() const
	{
#ifdef __IRR_COMPILE_WITH_SSE3
	    // xmm4-7 will now become columuns of B
	    __m128 xmm4 = rows[0].getAsRegister();
	    __m128 xmm5 = rows[1].getAsRegister();
	    __m128 xmm6 = rows[2].getAsRegister();
	    __m128 xmm7 = _mm_setzero_ps();
	    // g==0
	    __m128 xmm0 = _mm_unpacklo_ps(xmm4.xmm5);
	    __m128 xmm1 = _mm_unpacklo_ps(xmm6,xmm7); // (2x,g,2y,g)
	    __m128 xmm2 = _mm_unpackhi_ps(xmm4,xmm5);
	    __m128 xmm3 = _mm_unpackhi_ps(xmm6,xmm7); // (2z,g,2w,g)
	    xmm4 = _mm_movelh_ps(xmm1,xmm0); //(0x,1x,2x,g)
	    xmm5 = _mm_movehl_ps(xmm1,xmm0);
	    xmm6 = _mm_movelh_ps(xmm3,xmm2); //(0z,1z,,2z,g)

		// See http://www.robertblum.com/articles/2005/02/14/decomposing-matrices
		// We have to do the full calculation.
		xmm0 = _mm_mul_ps(xmm4,xmm4);// column 0 squared
		xmm1 = _mm_mul_ps(xmm5,xmm5);// column 1 squared
		xmm2 = _mm_mul_ps(xmm6,xmm6);// column 2 squared
		xmm4 = _mm_hadd_ps(xmm0,xmm1);
		xmm5 = _mm_hadd_ps(xmm2,xmm7);
		xmm6 = _mm_hadd_ps(xmm4,xmm5);
		return xmm6;
#elif defined(__IRR_COMPILE_WITH_SSE2)
#error "SSE2 version not implemented yet"
#endif
	}
	inline vectorSIMDf matrixSIMD4::getScale() const
	{
#ifdef __IRR_COMPILE_WITH_SSE3
		return getScaleSQ().getSquareRoot();
#elif defined(__IRR_COMPILE_WITH_SSE2)
#error "SSE2 version not implemented yet"
#endif
	}

/*
	inline matrixSIMD4& matrixSIMD4::setRotationDegrees( const vectorSIMDf& rotation )
	{
		return setRotationRadians( rotation * core::DEGTORAD );
	}


	inline matrixSIMD4& matrixSIMD4::setInverseRotationDegrees( const vectorSIMDf& rotation )
	{
		return setRotationRadians( rotation * (-core::DEGTORAD) );
	}

	inline matrixSIMD4& matrixSIMD4::setRotationRadians( const vectorSIMDf& rotation )
	{
		const float cr = cosf( rotation.X );
		const float sr = sinf( rotation.X );
		const float cp = cosf( rotation.Y );
		const float sp = sinf( rotation.Y );
		const float cy = cosf( rotation.Z );
		const float sy = sinf( rotation.Z );
		const float c = cosf( rotation.W );
		const float s = sinf( rotation.W );

		M[0] = ( cp*cy );
		M[1] = ( cp*sy );
		M[2] = ( -sp );

		const f64 srsp = sr*sp;
		const f64 crsp = cr*sp;

		M[4] = ( srsp*cy-cr*sy );
		M[5] = ( srsp*sy+cr*cy );
		M[6] = ( sr*cp );

		M[8] = ( crsp*cy+sr*sy );
		M[9] = ( crsp*sy-sr*cy );
		M[10] = ( cr*cp );
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}

	//! Sets matrix to rotation matrix of inverse angles given as parameters
	inline matrixSIMD4& matrixSIMD4::setInverseRotationRadians( const vectorSIMDf& rotation )
	{
		return setRotationRadians(-rotation);
	}


	//! Returns a rotation that is equivalent to that set by setRotationDegrees().
	/** This code was sent in by Chev.  Note that it does not necessarily return
	the *same* Euler angles as those set by setRotationDegrees(), but the rotation will
	be equivalent, i.e. will have the same result when used to rotate a vector or node. *
	inline vectorSIMDf matrixSIMD4::getRotationDegrees() const
	{
		return getRotationRadians()*core::RADTODEG;
	}

/*
	//! Sets matrix to rotation matrix defined by axis and angle, assuming LH rotation
	inline matrixSIMD4& matrixSIMD4::setRotationAxisRadians( const float &angle, const vectorSIMDf& axis )
	{
 		const f64 c = cos(angle);
		const f64 s = sin(angle);
		const f64 t = 1.0 - c;

		const f64 tx  = t * axis.X;
		const f64 ty  = t * axis.Y;
		const f64 tz  = t * axis.Z;

		const f64 sx  = s * axis.X;
		const f64 sy  = s * axis.Y;
		const f64 sz  = s * axis.Z;

		M[0] = (tx * axis.X + c);
		M[1] = (tx * axis.Y + sz);
		M[2] = (tx * axis.Z - sy);

		M[4] = (ty * axis.X - sz);
		M[5] = (ty * axis.Y + c);
		M[6] = (ty * axis.Z + sx);

		M[8]  = (tz * axis.X + sy);
		M[9]  = (tz * axis.Y - sx);
		M[10] = (tz * axis.Z + c);

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	/*
		check identity with epsilon
		solve floating range problems..
	*/
	inline bool matrixSIMD4::isIdentity() const
	{
#if defined ( USE_MATRIX_TEST )
		if (definitelyIdentityMatrix)
			return true;
#endif
        vector4db_SIMD tmp = (rows[0]!=vectorSIMDf(1.f,0.f,0.f,0.f))|(rows[1]!=vectorSIMDf(0.f,1.f,0.f,0.f))|(rows[2]!=vectorSIMDf(0.f,0.f,1.f,0.f))|(rows[3]!=vectorSIMDf(0.f,0.f,0.f,1.f));

		if (tmp.any())
			return false;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=true;
#endif
		return true;
	}


	/* Check orthogonality of matrix. */
	inline bool matrixSIMD4::isOrthogonal() const
	{
		//all of the column vectors have to be orthogonal to each other
		return ((*this)*(*this).getTransposed()).isIdentity();
	}


/*
	inline void matrixSIMD4::rotateVect( vector3df& vect ) const
	{
		vector3df tmp = vect;
		vect.X = tmp.X*M[0] + tmp.Y*M[4] + tmp.Z*M[8];
		vect.Y = tmp.X*M[1] + tmp.Y*M[5] + tmp.Z*M[9];
		vect.Z = tmp.X*M[2] + tmp.Y*M[6] + tmp.Z*M[10];
	}

	//! An alternate transform vector method, writing into a second vector
	inline void matrixSIMD4::rotateVect(core::vector3df& out, const core::vector3df& in) const
	{
		out.X = in.X*M[0] + in.Y*M[4] + in.Z*M[8];
		out.Y = in.X*M[1] + in.Y*M[5] + in.Z*M[9];
		out.Z = in.X*M[2] + in.Y*M[6] + in.Z*M[10];
	}

	//! An alternate transform vector method, writing into an array of 3 floats
	inline void matrixSIMD4::rotateVect(float *out, const core::vector3df& in) const
	{
		out[0] = in.X*M[0] + in.Y*M[4] + in.Z*M[8];
		out[1] = in.X*M[1] + in.Y*M[5] + in.Z*M[9];
		out[2] = in.X*M[2] + in.Y*M[6] + in.Z*M[10];
	}
*/

	inline void matrixSIMD4::transformVect( vectorSIMDf& vect) const
	{
        transformVect(vect,vect);
	}

	inline void matrixSIMD4::transformVect( vectorSIMDf& out, const vectorSIMDf& in) const
	{
	    transformVect(out.pointer,in);
	}


	inline void matrixSIMD4::transformVect(float *out, const vectorSIMDf &in) const
	{
	    __m128 xmm4 = in.getAsRegister();
	    __m128 xmm0 = _mm_mul_ps(rows[0].getAsRegister(),xmm4);
	    __m128 xmm1 = _mm_mul_ps(rows[1].getAsRegister(),xmm4);
	    __m128 xmm2 = _mm_mul_ps(rows[2].getAsRegister(),xmm4);
	    __m128 xmm3 = _mm_mul_ps(rows[3].getAsRegister(),xmm4);
	    xmm4 = _mm_hadd_ps(xmm2,xmm3);
	    xmm2 = _mm_hadd_ps(xmm0,xmm1);
	    _mm_store_ps(out,_mm_hadd_ps(xmm2,xmm4));
	}

/*
	//! Transforms a plane by this matrix
	inline void matrixSIMD4::transformPlane( core::plane3d<f32> &plane) const
	{
		core::plane3df temp;
		transformPlane(plane,temp);
		plane = temp;
	}

	//! Transforms a plane by this matrix
	inline void matrixSIMD4::transformPlane( const core::plane3d<f32> &in, core::plane3d<f32> &out) const
	{
		matrixSIMD4 transposedInverse(*this, EM4CONST_INVERSE);
        out.Normal.X = in.Normal.X*transposedInverse[0] + in.Normal.Y*transposedInverse[1] + in.Normal.Z*transposedInverse[2] + in.D*transposedInverse[3];
        out.Normal.Y = in.Normal.X*transposedInverse[4] + in.Normal.Y*transposedInverse[5] + in.Normal.Z*transposedInverse[6] + in.D*transposedInverse[7];
        out.Normal.Z = in.Normal.X*transposedInverse[8] + in.Normal.Y*transposedInverse[9] + in.Normal.Z*transposedInverse[10] + in.D*transposedInverse[11];
        out.D = in.Normal.X*transposedInverse[12] + in.Normal.Y*transposedInverse[13] + in.Normal.Z*transposedInverse[14] + in.D*transposedInverse[15];
	}

	//! Transforms a axis aligned bounding box
	inline void matrixSIMD4::transformBox(core::aabbox3d<f32>& box) const
	{
#if defined ( USE_MATRIX_TEST )
		if (isIdentity())
			return;
#endif

		transformVect(box.MinEdge);
		transformVect(box.MaxEdge);
		box.repair();
	}

	//! Transforms a axis aligned bounding box more accurately than transformBox()
	inline void matrixSIMD4::transformBoxEx(core::aabbox3d<f32>& box) const
	{
#if defined ( USE_MATRIX_TEST )
		if (isIdentity())
			return;
#endif

		const f32 Amin[3] = {box.MinEdge.X, box.MinEdge.Y, box.MinEdge.Z};
		const f32 Amax[3] = {box.MaxEdge.X, box.MaxEdge.Y, box.MaxEdge.Z};

		f32 Bmin[3];
		f32 Bmax[3];

		Bmin[0] = Bmax[0] = M[12];
		Bmin[1] = Bmax[1] = M[13];
		Bmin[2] = Bmax[2] = M[14];

		const matrixSIMD4 &m = *this;

		for (u32 i = 0; i < 3; ++i)
		{
			for (u32 j = 0; j < 3; ++j)
			{
				const f32 a = m(j,i) * Amin[j];
				const f32 b = m(j,i) * Amax[j];

				if (a < b)
				{
					Bmin[i] += a;
					Bmax[i] += b;
				}
				else
				{
					Bmin[i] += b;
					Bmax[i] += a;
				}
			}
		}

		box.MinEdge.X = Bmin[0];
		box.MinEdge.Y = Bmin[1];
		box.MinEdge.Z = Bmin[2];

		box.MaxEdge.X = Bmax[0];
		box.MaxEdge.Y = Bmax[1];
		box.MaxEdge.Z = Bmax[2];
	}

*/
	inline void matrixSIMD4::inverseTranslateVect( vectorSIMDf& vect ) const
	{
	    __m128 xmm0 = _mm_unpackhi_ps(rows[0].getAsRegister(),rows[1].getAsRegister()); // (0z,1z,0w,1w)
	    __m128 xmm1 = _mm_unpackhi_ps(rows[2].getAsRegister(),_mm_setzero_ps()); // (2z,3z,2w,3w)
	    __m128 xmm2 = _mm_movehl_ps(xmm1,xmm0);// (0w,1w,2w,3w)

	    vect -= xmm2;
	}

	inline void matrixSIMD4::translateVect( vector3df& vect ) const
	{
	    __m128 xmm0 = _mm_unpackhi_ps(rows[0].getAsRegister(),rows[1].getAsRegister()); // (0z,1z,0w,1w)
	    __m128 xmm1 = _mm_unpackhi_ps(rows[2].getAsRegister(),_mm_setzero_ps()); // (2z,3z,2w,3w)
	    __m128 xmm2 = _mm_movehl_ps(xmm1,xmm0);// (0w,1w,2w,3w)

	    vect += xmm2;
	}


	inline bool matrixSIMD4::getInverse(matrixSIMD4& out) const
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
        vector4db_SIMD isReasonable = (rows[3]==vectorSIMDf(0.f,0.f,0.f,1.f));
        vectorSIMDf determinant4;

        if (isReasonable.all())
        {
            // last row is 0,0,0,1 like in a sane 4x4 matrix used in games
            vectorSIMDf tmpA = rows[1].zxxw()*rows[2].yzyw();// (m(1, 2) * m(2, 1)
            vectorSIMDf tmpB = rows[1].yzyw()*rows[2].zxxw();// (m(1, 1) * m(2, 2))
            __m128 tmpC = tmpA-tmpB; //1st column of out matrix
            __m128 preDeterminant = rows[0]*tmpC;
            preDeterminant = _mm_hadd_ps(preDeterminant,preDeterminant); // (x+y,z+w,..)
            determinant4 = _mm_hadd_ps(preDeterminant,preDeterminant); //

            if (((uint32_t*)determinant4.pointer)[0]==0.f)
                return false;


            tmpA = rows[0].zxyw()*rows[2].yzxw();
            tmpB = rows[0].yzxw()*rows[2].zxyw();
            __m128 tmpD = tmpA-tmpB; // 2nd column of out matrix

            tmpA = rows[0].yzxw()*rows[1].zxyw();
            tmpB = rows[0].zxyw()*rows[1].yzxw();
            __m128 tmpE = tmpA-tmpB; // 3rd column of out matrix

            __m128 xmm0 = tmpC;
            __m128 xmm1 = tmpD;
            __m128 xmm2 = tmpE;
            __m128 xmm3 = _mm_setzero_ps();

            _MM_TRANSPOSE4_PS(xmm0,xmm1,xmm2,xmm3)

            __m128 xmm4 = getTranslation3D().getAsRegister();


            xmm0 = _mm_mul_ps(xmm0,xmm4); //out(0,3)
            xmm1 = _mm_mul_ps(xmm1,xmm4); //out(1,3)
            xmm2 = _mm_or_ps(_mm_mul_ps(xmm2,xmm4),_mm_castsi128_ps(_mm_set_epi32(0,-1,0,-1))); //out(2,3)

            xmm0 = _mm_hsub_ps(xmm0,xmm1); // C.x-D.x,E.x,C.y-D.y,E.y
            xmm1 = _mm_hsub_ps(xmm2,preDeterminant); // C.z-D.z,E.z,x+y-z-w,x+Y-z-w
            xmm2 = _mm_hsub_ps(xmm0,xmm1); // C.x-D.x-E.x,C.y-D.y-E.y,C.z-D.z-E.z,0
/*
            out(0, 3) = m(0, 3) * tmpC.x +
                        m(1, 3) * -tmpD.x +
                        m(2, 3) * -tmpE.x;

            out(1, 3) = m(0, 3) * tmpC.y +
                        m(1, 3) * -tmpD.y +
                        m(2, 3) * -tmpE.y;

            out(2, 3) = m(0, 3) * -tmpC.z +
                        m(1, 3) * -tmpD.z;
                        m(2, 3) * tmpE.z;
*/

            _MM_TRANSPOSE4_PS(tmpC,tmpD,tmpE,xmm2)
            out.rows[0] = tmpC;
            out.rows[1] = tmpD;
            out.rows[2] = tmpE;
            out.rows[3] = xmm2;

            tmpA = xmm1;
            out[15] = -tmpA.w;
        }
        else
        {
            /**
            out(0, 0) = m(1, 1) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2)) + m(1, 2) * (m(2, 3) * m(3, 1) - m(2, 1) * m(3, 3)) + m(1, 3) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1)));
            out(1, 0) = m(1, 2) * (m(2, 0) * m(3, 3) - m(2, 3) * m(3, 0)) + m(1, 3) * (m(2, 2) * m(3, 0) - m(2, 0) * m(3, 2)) + m(1, 0) * (m(2, 3) * m(3, 2) - m(2, 2) * m(3, 3)));
            out(2, 0) = m(1, 3) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0)) + m(1, 0) * (m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1)) + m(1, 1) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)));
            out(3, 0) = m(1, 0) * (m(2, 2) * m(3, 1) - m(2, 1) * m(3, 2)) + m(1, 1) * (m(2, 0) * m(3, 2) - m(2, 2) * m(3, 0)) + m(1, 2) * (m(2, 1) * m(3, 0) - m(2, 0) * m(3, 1)));

            out(0, 1) = (m(2, 1) * (m(0, 2) * m(3, 3) - m(0, 3) * m(3, 2)) + m(2, 2) * (m(0, 3) * m(3, 1) - m(0, 1) * m(3, 3)) + m(2, 3) * (m(0, 1) * m(3, 2) - m(0, 2) * m(3, 1)));
            out(1, 1) = (m(2, 2) * (m(0, 0) * m(3, 3) - m(0, 3) * m(3, 0)) + m(2, 3) * (m(0, 2) * m(3, 0) - m(0, 0) * m(3, 2)) + m(2, 0) * (m(0, 3) * m(3, 2) - m(0, 2) * m(3, 3)));
            out(2, 1) = (m(2, 3) * (m(0, 0) * m(3, 1) - m(0, 1) * m(3, 0)) + m(2, 0) * (m(0, 1) * m(3, 3) - m(0, 3) * m(3, 1)) + m(2, 1) * (m(0, 3) * m(3, 0) - m(0, 0) * m(3, 3)));
            out(3, 1) = (m(2, 0) * (m(0, 2) * m(3, 1) - m(0, 1) * m(3, 2)) + m(2, 1) * (m(0, 0) * m(3, 2) - m(0, 2) * m(3, 0)) + m(2, 2) * (m(0, 1) * m(3, 0) - m(0, 0) * m(3, 1)));

            out(0, 2) = (m(3, 1) * (m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2)) + m(3, 2) * (m(0, 3) * m(1, 1) - m(0, 1) * m(1, 3)) + m(3, 3) * (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)));
            out(1, 2) = (m(3, 2) * (m(0, 0) * m(1, 3) - m(0, 3) * m(1, 0)) + m(3, 3) * (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) + m(3, 0) * (m(0, 3) * m(1, 2) - m(0, 2) * m(1, 3)));
            out(2, 2) = (m(3, 3) * (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) + m(3, 0) * (m(0, 1) * m(1, 3) - m(0, 3) * m(1, 1)) + m(3, 1) * (m(0, 3) * m(1, 0) - m(0, 0) * m(1, 3)));
            out(3, 2) = (m(3, 0) * (m(0, 2) * m(1, 1) - m(0, 1) * m(1, 2)) + m(3, 1) * (m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)) + m(3, 2) * (m(0, 1) * m(1, 0) - m(0, 0) * m(1, 1)));

            out(0, 3) = (m(0, 1) * (m(1, 3) * m(2, 2) - m(1, 2) * m(2, 3)) + m(0, 2) * (m(1, 1) * m(2, 3) - m(1, 3) * m(2, 1)) + m(0, 3) * (m(1, 2) * m(2, 1) - m(1, 1) * m(2, 2)));
            out(1, 3) = (m(0, 2) * (m(1, 3) * m(2, 0) - m(1, 0) * m(2, 3)) + m(0, 3) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) + m(0, 0) * (m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2)));
            out(2, 3) = (m(0, 3) * (m(1, 1) * m(2, 0) - m(1, 0) * m(2, 1)) + m(0, 0) * (m(1, 3) * m(2, 1) - m(1, 1) * m(2, 3)) + m(0, 1) * (m(1, 0) * m(2, 3) - m(1, 3) * m(2, 0)));
            out(3, 3) = (m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) + m(0, 1) * (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)));
            **/
            vectorSIMDf tmpA = rows[2].zxxz();
            vectorSIMDf tmpB = rows[3].wwyy();
            vectorSIMDf tmpC = rows[2].wwyy();
            vectorSIMDf tmpD = rows[3].zxxz();
            vectorSIMDf tmpE = rows[2].wzyx();
            vectorSIMDf tmpF = rows[3].yxwz();
            vectorSIMDf tmpG = rows[2].yxwz();
            vectorSIMDf tmpH = rows[3].wzyx();
            vectorSIMDf tmpI = rows[2].ywwy();
            vectorSIMDf tmpJ = rows[3].zzxx();
            vectorSIMDf tmpK = rows[2].zzxx();
            vectorSIMDf tmpL = rows[3].ywwy();
            __m128 xmm0 = (rows[1].yzwx()*(tmpA*tmpB-tmpC*tmpD)+rows[1].zwxy()*(tmpE*tmpF-tmpG*tmpH)+rows[1].wxyz()*(tmpI*tmpJ-tmpK*tmpL)).getAsRegister();

            determinant4 = rows[0].dotProduct(xmm0);
            if (((uint32_t*)determinant4.pointer)[0]==0.f)
                return false;

            vectorSIMDf tmpM = rows[0].zxxz();
            vectorSIMDf tmpN = rows[0].wwyy();
            vectorSIMDf tmpO = rows[0].wzyx();
            vectorSIMDf tmpP = rows[0].yxwz();
            vectorSIMDf tmpQ = rows[0].ywwy();
            vectorSIMDf tmpR = rows[0].zzxx();
            __m128 xmm1 = (rows[2].yzwx()*(tmpM*tmpB-tmpN*tmpD)+rows[2].zwxy()*(tmpO*tmpF-tmpP*tmpH)+rows[2].wxyz()*(tmpQ*tmpJ-tmpR*tmpL)).getAsRegister();
            vectorSIMDf tmpS = rows[1].wwyy();
            vectorSIMDf tmpT = rows[1].zxxz();
            vectorSIMDf tmpU = rows[1].yxwz();
            vectorSIMDf tmpV = rows[1].wzyx();
            vectorSIMDf tmpW = rows[1].zzxx();
            vectorSIMDf tmpX = rows[1].ywwy();
            __m128 xmm2 = (rows[3].yzwx()*(tmpM*tmpS-tmpN*tmpT)+rows[3].zwxy()*(tmpO*tmpU-tmpP*tmpV)+rows[3].wxyz()*(tmpQ*tmpW-tmpR*tmpX)).getAsRegister();
            __m128 xmm3 = (rows[0].yzwx()*(tmpS*tmpA-tmpT*tmpC)+rows[0].zwxy()*(tmpU*tmpE-tmpV*tmpG)+rows[0].wxyz()*(tmpW*tmpI-tmpX*tmpK)).getAsRegister();


            _MM_TRANSPOSE4_PS(xmm0,xmm1,xmm2,xmm3)
            out.rows[0] = xmm0;
            out.rows[1] = xmm1;
            out.rows[2] = xmm2;
            out.rows[3] = xmm3;
        }


        __m128 xmm0 = _mm_rcp_ps(determinant4.getAsRegister());
        out.rows[0] *= xmm0;
        out.rows[1] *= xmm0;
        out.rows[2] *= xmm0;
        out.rows[3] *= xmm0;

#if defined ( USE_MATRIX_TEST )
		out.definitelyIdentityMatrix = false;
#endif
		return true;
	}

/*
	//! Inverts a primitive matrix which only contains a translation and a rotation
	//! \param out: where result matrix is written to.
	inline bool matrixSIMD4::getInversePrimitive ( matrixSIMD4& out ) const
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

		out.M[12] = -(M[12]*M[0] + M[13]*M[1] + M[14]*M[2]);
		out.M[13] = -(M[12]*M[4] + M[13]*M[5] + M[14]*M[6]);
		out.M[14] = -(M[12]*M[8] + M[13]*M[9] + M[14]*M[10]);
		out.M[15] = 1;

#if defined ( USE_MATRIX_TEST )
		out.definitelyIdentityMatrix = definitelyIdentityMatrix;
#endif
		return true;
	}

	//!
	inline bool matrixSIMD4::makeInverse()
	{
#if defined ( USE_MATRIX_TEST )
		if (definitelyIdentityMatrix)
			return true;
#endif
		matrixSIMD4 temp ( EM4CONST_NOTHING );

		if (getInverse(temp))
		{
			*this = temp;
			return true;
		}

		return false;
	}
*/

	inline matrixSIMD4& matrixSIMD4::operator=(const matrixSIMD4 &other)
	{
		_mm_store_ps(rows[0].pointer,other.rows[0].getAsRegister());
		_mm_store_ps(rows[1].pointer,other.rows[1].getAsRegister());
		_mm_store_ps(rows[2].pointer,other.rows[2].getAsRegister());
		_mm_store_ps(rows[3].pointer,other.rows[3].getAsRegister());
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=other.definitelyIdentityMatrix;
#endif
		return *this;
	}


	inline matrixSIMD4& matrixSIMD4::operator=(const float& scalar)
	{
	    __m128 xmm0 = _mm_load_ps1(&scalar);
		_mm_store_ps(rows[0].pointer,xmm0);
		_mm_store_ps(rows[1].pointer,xmm0);
		_mm_store_ps(rows[2].pointer,xmm0);
		_mm_store_ps(rows[3].pointer,xmm0);

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	inline bool matrixSIMD4::operator==(const matrixSIMD4 &other) const
	{
#if defined ( USE_MATRIX_TEST )
		if (definitelyIdentityMatrix && other.definitelyIdentityMatrix)
			return true;
#endif

        return !((*this)!=other);
	}


	inline bool matrixSIMD4::operator!=(const matrixSIMD4 &other) const
	{
        return ((rows[0]!=other.rows[0])|(rows[1]!=other.rows[1])|(rows[2]!=other.rows[2])|(rows[3]!=other.rows[3])).any();
	}

/*
	// Builds a right-handed perspective projection matrix based on a field of view
	inline matrixSIMD4& matrixSIMD4::buildProjectionMatrixPerspectiveFovRH(
			f32 fieldOfViewRadians, f32 aspectRatio, f32 zNear, f32 zFar)
	{
		const f32 h = reciprocal(tan(fieldOfViewRadians*0.5));
		_IRR_DEBUG_BREAK_IF(aspectRatio==0.f); //divide by zero
		const float w = h / aspectRatio;

		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = w;
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = h;
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (zFar/(zNear-zFar)); // DirectX version
//		M[10] = (zFar+zNear/(zNear-zFar)); // OpenGL version
		M[11] = -1;

		M[12] = 0;
		M[13] = 0;
		M[14] = (zNear*zFar/(zNear-zFar)); // DirectX version
//		M[14] = (2.0f*zNear*zFar/(zNear-zFar)); // OpenGL version
		M[15] = 0;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a left-handed perspective projection matrix based on a field of view
	inline matrixSIMD4& matrixSIMD4::buildProjectionMatrixPerspectiveFovLH(
			f32 fieldOfViewRadians, f32 aspectRatio, f32 zNear, f32 zFar)
	{
		const f32 h = reciprocal(tan(fieldOfViewRadians*0.5));
		_IRR_DEBUG_BREAK_IF(aspectRatio==0.f); //divide by zero
		const float w = (h / aspectRatio);

		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = w;
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = h;
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (zFar/(zFar-zNear));
		M[11] = 1;

		M[12] = 0;
		M[13] = 0;
		M[14] = (-zNear*zFar/(zFar-zNear));
		M[15] = 0;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a left-handed perspective projection matrix based on a field of view, with far plane culling at infinity
	inline matrixSIMD4& matrixSIMD4::buildProjectionMatrixPerspectiveFovInfinityLH(
			f32 fieldOfViewRadians, f32 aspectRatio, f32 zNear, f32 epsilon)
	{
		const f32 h = reciprocal(tan(fieldOfViewRadians*0.5));
		_IRR_DEBUG_BREAK_IF(aspectRatio==0.f); //divide by zero
		const float w = h / aspectRatio;

		M[0] = w;
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = h;
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (1.f-epsilon);
		M[11] = 1;

		M[12] = 0;
		M[13] = 0;
		M[14] = (zNear*(epsilon-1.f));
		M[15] = 0;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a left-handed orthogonal projection matrix.
	inline matrixSIMD4& matrixSIMD4::buildProjectionMatrixOrthoLH(
			f32 widthOfViewVolume, f32 heightOfViewVolume, f32 zNear, f32 zFar)
	{
		_IRR_DEBUG_BREAK_IF(widthOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(heightOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = (2/widthOfViewVolume);
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = (2/heightOfViewVolume);
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (1/(zFar-zNear));
		M[11] = 0;

		M[12] = 0;
		M[13] = 0;
		M[14] = (zNear/(zNear-zFar));
		M[15] = 1;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a right-handed orthogonal projection matrix.
	inline matrixSIMD4& matrixSIMD4::buildProjectionMatrixOrthoRH(
			f32 widthOfViewVolume, f32 heightOfViewVolume, f32 zNear, f32 zFar)
	{
		_IRR_DEBUG_BREAK_IF(widthOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(heightOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = (2/widthOfViewVolume);
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = (2/heightOfViewVolume);
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (1/(zNear-zFar));
		M[11] = 0;

		M[12] = 0;
		M[13] = 0;
		M[14] = (zNear/(zNear-zFar));
		M[15] = 1;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a right-handed perspective projection matrix.
	inline matrixSIMD4& matrixSIMD4::buildProjectionMatrixPerspectiveRH(
			f32 widthOfViewVolume, f32 heightOfViewVolume, f32 zNear, f32 zFar)
	{
		_IRR_DEBUG_BREAK_IF(widthOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(heightOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = (2*zNear/widthOfViewVolume);
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = (2*zNear/heightOfViewVolume);
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (zFar/(zNear-zFar));
		M[11] = -1;

		M[12] = 0;
		M[13] = 0;
		M[14] = (zNear*zFar/(zNear-zFar));
		M[15] = 0;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a left-handed perspective projection matrix.
	inline matrixSIMD4& matrixSIMD4::buildProjectionMatrixPerspectiveLH(
			f32 widthOfViewVolume, f32 heightOfViewVolume, f32 zNear, f32 zFar)
	{
		_IRR_DEBUG_BREAK_IF(widthOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(heightOfViewVolume==0.f); //divide by zero
		_IRR_DEBUG_BREAK_IF(zNear==zFar); //divide by zero
		M[0] = (2*zNear/widthOfViewVolume);
		M[1] = 0;
		M[2] = 0;
		M[3] = 0;

		M[4] = 0;
		M[5] = (2*zNear/heightOfViewVolume);
		M[6] = 0;
		M[7] = 0;

		M[8] = 0;
		M[9] = 0;
		M[10] = (zFar/(zFar-zNear));
		M[11] = 1;

		M[12] = 0;
		M[13] = 0;
		M[14] = (zNear*zFar/(zNear-zFar));
		M[15] = 0;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a matrix that flattens geometry into a plane.
	inline matrixSIMD4& matrixSIMD4::buildShadowMatrix(const core::vector3df& light, core::plane3df plane, f32 point)
	{
		plane.Normal.normalize();
		const f32 d = plane.Normal.dotProduct(light);

		M[ 0] = (-plane.Normal.X * light.X + d);
		M[ 1] = (-plane.Normal.X * light.Y);
		M[ 2] = (-plane.Normal.X * light.Z);
		M[ 3] = (-plane.Normal.X * point);

		M[ 4] = (-plane.Normal.Y * light.X);
		M[ 5] = (-plane.Normal.Y * light.Y + d);
		M[ 6] = (-plane.Normal.Y * light.Z);
		M[ 7] = (-plane.Normal.Y * point);

		M[ 8] = (-plane.Normal.Z * light.X);
		M[ 9] = (-plane.Normal.Z * light.Y);
		M[10] = (-plane.Normal.Z * light.Z + d);
		M[11] = (-plane.Normal.Z * point);

		M[12] = (-plane.D * light.X);
		M[13] = (-plane.D * light.Y);
		M[14] = (-plane.D * light.Z);
		M[15] = (-plane.D * point + d);
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}

	// Builds a left-handed look-at matrix.
	inline matrixSIMD4& matrixSIMD4::buildCameraLookAtMatrixLH(
				const vector3df& position,
				const vector3df& target,
				const vector3df& upVector)
	{
		vector3df zaxis = target - position;
		zaxis.normalize();

		vector3df xaxis = upVector.crossProduct(zaxis);
		xaxis.normalize();

		vector3df yaxis = zaxis.crossProduct(xaxis);

		M[0] = xaxis.X;
		M[1] = yaxis.X;
		M[2] = zaxis.X;
		M[3] = 0;

		M[4] = xaxis.Y;
		M[5] = yaxis.Y;
		M[6] = zaxis.Y;
		M[7] = 0;

		M[8] = xaxis.Z;
		M[9] = yaxis.Z;
		M[10] =zaxis.Z;
		M[11] = 0;

		M[12] = -xaxis.dotProduct(position);
		M[13] = -yaxis.dotProduct(position);
		M[14] = -zaxis.dotProduct(position);
		M[15] = 1;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// Builds a right-handed look-at matrix.
	inline matrixSIMD4& matrixSIMD4::buildCameraLookAtMatrixRH(
				const vector3df& position,
				const vector3df& target,
				const vector3df& upVector)
	{
		vector3df zaxis = position - target;
		zaxis.normalize();

		vector3df xaxis = upVector.crossProduct(zaxis);
		xaxis.normalize();

		vector3df yaxis = zaxis.crossProduct(xaxis);

		M[0] = xaxis.X;
		M[1] = yaxis.X;
		M[2] = zaxis.X;
		M[3] = 0;

		M[4] = xaxis.Y;
		M[5] = yaxis.Y;
		M[6] = zaxis.Y;
		M[7] = 0;

		M[8] = xaxis.Z;
		M[9] = yaxis.Z;
		M[10] = zaxis.Z;
		M[11] = 0;

		M[12] = -xaxis.dotProduct(position);
		M[13] = -yaxis.dotProduct(position);
		M[14] = -zaxis.dotProduct(position);
		M[15] = 1;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}
*/

	// creates a new matrix as interpolated matrix from this and the passed one.
	inline matrixSIMD4 matrixSIMD4::interpolate(const matrixSIMD4& b, const float &factor) const
	{
		matrixSIMD4 mat ( EM4CONST_NOTHING );

		mat.rows[0] = vectorSIMDf::mix(this->rows[0],b.rows[0],factor);
		mat.rows[1] = vectorSIMDf::mix(this->rows[1],b.rows[1],factor);
		mat.rows[2] = vectorSIMDf::mix(this->rows[2],b.rows[2],factor);
		mat.rows[3] = vectorSIMDf::mix(this->rows[3],b.rows[3],factor);
		return mat;
	}


	// returns transposed matrix
	inline matrixSIMD4 matrixSIMD4::getTransposed() const
	{
		matrixSIMD4 t ( EM4CONST_NOTHING );
		getTransposed ( t );
		return t;
	}


	// returns transposed matrix
	inline void matrixSIMD4::getTransposed( matrixSIMD4& o ) const
	{
	    __m128 xmm0 = rows[0].getAsRegister();
	    __m128 xmm1 = rows[1].getAsRegister();
	    __m128 xmm2 = rows[2].getAsRegister();
	    __m128 xmm3 = rows[3].getAsRegister();
	    _MM_TRANSPOSE4_PS(xmm0,xmm1,xmm2,xmm3)
	    _mm_store_ps((float*)o.rows,xmm0);
	    _mm_store_ps((float*)(o.rows+1),xmm1);
	    _mm_store_ps((float*)(o.rows+2),xmm2);
	    _mm_store_ps((float*)(o.rows+3),xmm3);
#if defined ( USE_MATRIX_TEST )
		o.definitelyIdentityMatrix=definitelyIdentityMatrix;
#endif
	}

/*
	// used to scale <-1,-1><1,1> to viewport
	inline matrixSIMD4& matrixSIMD4::buildNDCToDCMatrix( const core::rect<s32>& viewport, f32 zScale)
	{
		const f32 scaleX = (viewport.getWidth() - 0.75f ) * 0.5f;
		const f32 scaleY = -(viewport.getHeight() - 0.75f ) * 0.5f;

		const f32 dx = -0.5f + ( (viewport.UpperLeftCorner.X + viewport.LowerRightCorner.X ) * 0.5f );
		const f32 dy = -0.5f + ( (viewport.UpperLeftCorner.Y + viewport.LowerRightCorner.Y ) * 0.5f );

		makeIdentity();
		M[12] = dx;
		M[13] = dy;
		return setScale(core::vectorSIMDf(scaleX, scaleY, zScale));
	}

	//! Builds a matrix that rotates from one vector to another
	/** \param from: vector to rotate from
	\param to: vector to rotate to

		http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
	 *
	inline matrixSIMD4& matrixSIMD4::buildRotateFromTo(const core::vector3df& from, const core::vector3df& to)
	{
		// unit vectors
		core::vector3df f(from);
		core::vector3df t(to);
		f.normalize();
		t.normalize();

		// axis multiplication by sin
		core::vector3df vs(t.crossProduct(f));

		// axis of rotation
		core::vector3df v(vs);
		v.normalize();

		// cosinus angle
		float ca = f.dotProduct(t);

		core::vector3df vt(v * (1 - ca));

		M[0] = vt.X * v.X + ca;
		M[5] = vt.Y * v.Y + ca;
		M[10] = vt.Z * v.Z + ca;

		vt.X *= v.Y;
		vt.Z *= v.X;
		vt.Y *= v.Z;

		M[1] = vt.X - vs.Z;
		M[2] = vt.Z + vs.Y;
		M[3] = 0;

		M[4] = vt.X + vs.Z;
		M[6] = vt.Y - vs.X;
		M[7] = 0;

		M[8] = vt.Z - vs.Y;
		M[9] = vt.Y + vs.X;
		M[11] = 0;

		M[12] = 0;
		M[13] = 0;
		M[14] = 0;
		M[15] = 1;

		return *this;
	}

	//! Builds a matrix which rotates a source vector to a look vector over an arbitrary axis
	/** \param camPos: viewer position in world coord
	\param center: object position in world-coord, rotation pivot
	\param translation: object final translation from center
	\param axis: axis to rotate about
	\param from: source vector to rotate from
	 *
	inline void matrixSIMD4::buildAxisAlignedBillboard(
				const core::vector3df& camPos,
				const core::vector3df& center,
				const core::vector3df& translation,
				const core::vector3df& axis,
				const core::vector3df& from)
	{
		// axis of rotation
		core::vector3df up = axis;
		up.normalize();
		const core::vector3df forward = (camPos - center).normalize();
		const core::vector3df right = up.crossProduct(forward).normalize();

		// correct look vector
		const core::vector3df look = right.crossProduct(up);

		// rotate from to
		// axis multiplication by sin
		const core::vector3df vs = look.crossProduct(from);

		// cosinus angle
		const f32 ca = from.dotProduct(look);

		core::vector3df vt(up * (1.f - ca));

		M[0] = (vt.X * up.X + ca);
		M[5] = (vt.Y * up.Y + ca);
		M[10] =(vt.Z * up.Z + ca);

		vt.X *= up.Y;
		vt.Z *= up.X;
		vt.Y *= up.Z;

		M[1] = (vt.X - vs.Z);
		M[2] = (vt.Z + vs.Y);
		M[3] = 0;

		M[4] = (vt.X + vs.Z);
		M[6] = (vt.Y - vs.X);
		M[7] = 0;

		M[8] = (vt.Z - vs.Y);
		M[9] = (vt.Y + vs.X);
		M[11] = 0;

		setRotationCenter(center, translation);
	}


	//! Builds a combined matrix which translate to a center before rotation and translate afterwards
	inline void matrixSIMD4::setRotationCenter(const core::vector3df& center, const core::vector3df& translation)
	{
		M[12] = -M[0]*center.X - M[4]*center.Y - M[8]*center.Z + (center.X - translation.X );
		M[13] = -M[1]*center.X - M[5]*center.Y - M[9]*center.Z + (center.Y - translation.Y );
		M[14] = -M[2]*center.X - M[6]*center.Y - M[10]*center.Z + (center.Z - translation.Z );
		M[15] = 1.0;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
	}



	inline matrixSIMD4& matrixSIMD4::buildTextureTransform( f32 rotateRad,
			const core::vector2df &rotatecenter,
			const core::vector2df &translate,
			const core::vector2df &scale)
	{
		const f32 c = cosf(rotateRad);
		const f32 s = sinf(rotateRad);

		M[0] = (c * scale.X);
		M[1] = (s * scale.Y);
		M[2] = 0;
		M[3] = 0;

		M[4] = (-s * scale.X);
		M[5] = (c * scale.Y);
		M[6] = 0;
		M[7] = 0;

		M[8] = (c * scale.X * rotatecenter.X + -s * rotatecenter.Y + translate.X);
		M[9] = (s * scale.Y * rotatecenter.X +  c * rotatecenter.Y + translate.Y);
		M[10] = 1;
		M[11] = 0;

		M[12] = 0;
		M[13] = 0;
		M[14] = 0;
		M[15] = 1;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// rotate about z axis, center ( 0.5, 0.5 )
	inline matrixSIMD4& matrixSIMD4::setTextureRotationCenter( f32 rotateRad )
	{
		const f32 c = cosf(rotateRad);
		const f32 s = sinf(rotateRad);
		M[0] = c;
		M[1] = s;

		M[4] = -s;
		M[5] = c;

		M[8] = (0.5f * ( s - c) + 0.5f);
		M[9] = (-0.5f * ( s + c) + 0.5f);

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (rotateRad==0.0f);
#endif
		return *this;
	}


	inline matrixSIMD4& matrixSIMD4::setTextureTranslate ( f32 x, f32 y )
	{
		M[8] = x;
		M[9] = y;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (x==0.0f) && (y==0.0f);
#endif
		return *this;
	}


	inline matrixSIMD4& matrixSIMD4::setTextureTranslateTransposed ( f32 x, f32 y )
	{
		M[2] = x;
		M[6] = y;

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (x==0.0f) && (y==0.0f) ;
#endif
		return *this;
	}

	inline matrixSIMD4& matrixSIMD4::setTextureScale ( f32 sx, f32 sy )
	{
		M[0] = sx;
		M[5] = sy;
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (sx==1.0f) && (sy==1.0f);
#endif
		return *this;
	}


	inline matrixSIMD4& matrixSIMD4::setTextureScaleCenter( f32 sx, f32 sy )
	{
		M[0] = sx;
		M[5] = sy;
		M[8] = (0.5f - 0.5f * sx);
		M[9] = (0.5f - 0.5f * sy);

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = definitelyIdentityMatrix && (sx==1.0f) && (sy==1.0f);
#endif
		return *this;
	}


	// sets all matrix data members at once
	inline matrixSIMD4& matrixSIMD4::setM(const float* data)
	{
		memcpy(M,data, 16*sizeof(float));

#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix=false;
#endif
		return *this;
	}


	// sets if the matrix is definitely identity matrix
	inline void matrixSIMD4::setDefinitelyIdentityMatrix( bool isDefinitelyIdentityMatrix)
	{
#if defined ( USE_MATRIX_TEST )
		definitelyIdentityMatrix = isDefinitelyIdentityMatrix;
#endif
	}


	// gets if the matrix is definitely identity matrix
	inline bool matrixSIMD4::getDefinitelyIdentityMatrix() const
	{
#if defined ( USE_MATRIX_TEST )
		return definitelyIdentityMatrix;
#else
		return false;
#endif
	}


	//! Compare two matrices using the equal method
	inline bool matrixSIMD4::equals(const core::matrixSIMD4& other, const float tolerance) const
	{
#if defined ( USE_MATRIX_TEST )
		if (definitelyIdentityMatrix && other.definitelyIdentityMatrix)
			return true;
#endif
		for (s32 i = 0; i < 16; ++i)
			if (!core::equals(M[i],other.M[i], tolerance))
				return false;

		return true;
	}


	// Multiply by scalar.
	inline matrixSIMD4 operator*(const float scalar, const matrixSIMD4& mat)
	{
		return mat*scalar;
	}*/


	//! global const identity matrix
	IRRLICHT_API extern const matrixSIMD4 IdentityMatrix;

} // end namespace core
} // end namespace irr

#endif
#endif

