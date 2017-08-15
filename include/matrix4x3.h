
// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MATRIX_4X3_H_INCLUDED__
#define __IRR_MATRIX_4X3_H_INCLUDED__

#include "irrMath.h"
#include "vector3d.h"
#include "vectorSIMD.h"
#include "plane3d.h"
#include "aabbox3d.h"
#include "rect.h"
#include "matrix4.h"



namespace irr
{
namespace core
{

    class matrix4x3;

    /** Calculate a*b */
    matrix4x3 concatenateBFollowedByA(const matrix4x3& other_a,const matrix4x3& other_b );

    matrix4x3 concatenatePreciselyBFollowedByA(const matrix4x3& other_a,const matrix4x3& other_b );

    matrix4 concatenateBFollowedByA(const matrix4& other_a,const matrix4x3& other_b );

    matrix4 concatenatePreciselyBFollowedByA(const matrix4& other_a,const matrix4x3& other_b );

	class matrix4x3
	{
		public:

			//! Constructor Flags
			enum eConstructor
			{
				EM4CONST_NOTHING = 0,
				EM4CONST_COPY,
				EM4CONST_IDENTITY,
				//EM4CONST_TRANSPOSED,
				EM4CONST_INVERSE=4,
				//EM4CONST_INVERSE_TRANSPOSED
			};

			//! Default constructor
			/** \param constructor Choose the initialization style */
			inline matrix4x3( eConstructor constructor = EM4CONST_IDENTITY )
            {
                switch ( constructor )
                {
                    case EM4CONST_NOTHING:
                    case EM4CONST_COPY:
                        break;
                    case EM4CONST_IDENTITY:
                    case EM4CONST_INVERSE:
                    default:
                        column[0].set(1.f,0.f,0.f);
                        column[1].set(0.f,1.f,0.f);
                        column[2].set(0.f,0.f,1.f);
                        column[3].set(0.f,0.f,0.f);
                        break;
                }
            }
			//! Copy constructor
			/** \param other Other matrix to copy from
			\param constructor Choose the initialization style */
			inline matrix4x3(const matrix4x3& other, eConstructor constructor = EM4CONST_COPY)
			{
                switch ( constructor )
                {
                    case EM4CONST_IDENTITY:
                        column[0].set(1.f,0.f,0.f);
                        column[1].set(0.f,1.f,0.f);
                        column[2].set(0.f,0.f,1.f);
                        column[3].set(0.f,0.f,0.f);
                        break;
                    case EM4CONST_COPY:
                        *this = other;
                        break;
                    case EM4CONST_INVERSE:
                        if (!other.getInverse(*this))
                            memset(column, 0, 12*sizeof(float));
                        break;
                    default:
                        break;
                }
			}

			inline matrix4 convertTo4x4() const
			{
			    matrix4 retval;
			    reinterpret_cast<vector3df*>(retval.pointer()+0)[0] = column[0];
			    reinterpret_cast<vector3df*>(retval.pointer()+4)[0] = column[1];
			    reinterpret_cast<vector3df*>(retval.pointer()+8)[0] = column[2];
			    reinterpret_cast<vector3df*>(retval.pointer()+12)[0] = column[3];
				retval[15] = 1.f;
			    return retval;
			}

			//! Simple operator for directly accessing every element of the matrix.
			inline float& operator()(const size_t &i, const size_t &j) {return reinterpret_cast<float*>(column+j)[i];}

			//! Simple operator for directly accessing every element of the matrix.
			inline const float& operator()(const size_t& i, const size_t& j) const { return reinterpret_cast<const float*>(column+j)[i]; }
/*
			//! Simple operator for linearly accessing every element of the matrix.
			inline float& operator[](const size_t& index) {return reinterpret_cast<float*>(column)[index];}

			//! Simple operator for linearly accessing every element of the matrix.
			inline const float& operator[](const size_t& index) const {return reinterpret_cast<const float*>(column)[index];}
*/
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

			//! apply this transformation before other (i.e. this==world, other==viewproj, gl_WorldViewProj = world.concatenateBefore(viewproj) )
            inline matrix4x3& concatenateBefore(const matrix4x3& other)
            {
                *this = concatenateBFollowedByA(other,*this);
                return *this;
            }
            inline matrix4x3& concatenatePreciselyBefore(const matrix4x3& other)
            {
                *this = concatenatePreciselyBFollowedByA(other,*this);
                return *this;
            }

			//! apply this transformation after other (i.e. this==proj, other==view, gl_ViewProj = proj.concatenateAfter(view) )
            inline matrix4x3& concatenateAfter(const matrix4x3& other)
            {
                *this = concatenateBFollowedByA(*this,other);
                return *this;
            }
            inline matrix4x3& concatenatePreciselyAfter(const matrix4x3& other)
            {
                *this = concatenatePreciselyBFollowedByA(*this,other);
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
                return setRotationRadians( rotation * core::DEGTORAD );
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

			inline void transformVect(float *in_out) const
			{
			    reinterpret_cast<vector3df*>(in_out)[0] = column[0]*in_out[0]+column[1]*in_out[1]+column[2]*in_out[2]+column[3];
			}

			inline void transformVect(float *out, const float * in) const
			{
			    reinterpret_cast<vector3df*>(out)[0] = column[0]*in[0]+column[1]*in[1]+column[2]*in[2]+column[3];
			}

			inline void pseudoMulWith4x1(float *in_out) const
			{
			    reinterpret_cast<vector3df*>(in_out)[0] = column[0]*in_out[0]+column[1]*in_out[1]+column[2]*in_out[2]+column[3];
			    in_out[3] = 1.f;
			}

			inline void pseudoMulWith4x1(float *out, const float * in) const
			{
			    reinterpret_cast<vector3df*>(out)[0] = column[0]*in[0]+column[1]*in[1]+column[2]*in[2]+column[3];
			    out[3] = 1.f;
			}

			inline void mulSub3x3With3x1(float *in_out) const
			{
			    reinterpret_cast<vector3df*>(in_out)[0] = column[0]*in_out[0]+column[1]*in_out[1]+column[2]*in_out[2];
			}

			inline void mulSub3x3With3x1(float *out, const float * in) const
			{
			    reinterpret_cast<vector3df*>(out)[0] = column[0]*in[0]+column[1]*in[1]+column[2]*in[2];
			}

			//! Transforms a plane by this matrix
			inline void transformPlane( core::plane3d<float> &plane) const
			{
                core::plane3df temp;
                transformPlane(plane,temp);
                plane = temp;
			}

			//! Transforms a plane by this matrix
			// (N,D).(x,1) = 0
			// N.(Mx) + D = 0
			inline void transformPlane( const core::plane3d<float> &in, core::plane3d<float> &out) const
            {
                matrix4x3 inverse;
                getInverse(inverse);

                out.Normal.X = in.Normal.dotProduct(inverse.column[0]);
                out.Normal.Y = in.Normal.dotProduct(inverse.column[1]);
                out.Normal.Z = in.Normal.dotProduct(inverse.column[2]);

                out.D = in.Normal.dotProduct(inverse.column[3]) + in.D;
            }


			//! Transforms a axis aligned bounding box
			inline void transformBoxEx(core::aabbox3d<float>& box) const
			{
			    core::aabbox3df tmpBox;
                tmpBox.MinEdge.X = column[0].X*(column[0].X<0.f ? box.MaxEdge.X:box.MinEdge.X)+column[1].X*(column[1].X<0.f ? box.MaxEdge.Y:box.MinEdge.Y)+column[2].X*(column[2].X<0.f ? box.MaxEdge.Z:box.MinEdge.Z);
                tmpBox.MinEdge.Y = column[0].Y*(column[0].Y<0.f ? box.MaxEdge.X:box.MinEdge.X)+column[1].Y*(column[1].Y<0.f ? box.MaxEdge.Y:box.MinEdge.Y)+column[2].Y*(column[2].Y<0.f ? box.MaxEdge.Z:box.MinEdge.Z);
                tmpBox.MinEdge.Z = column[0].Z*(column[0].Z<0.f ? box.MaxEdge.X:box.MinEdge.X)+column[1].Z*(column[1].Z<0.f ? box.MaxEdge.Y:box.MinEdge.Y)+column[2].Z*(column[2].Z<0.f ? box.MaxEdge.Z:box.MinEdge.Z);
                tmpBox.MaxEdge.X = column[0].X*(column[0].X<0.f ? box.MinEdge.X:box.MaxEdge.X)+column[1].X*(column[1].X<0.f ? box.MinEdge.Y:box.MaxEdge.Y)+column[2].X*(column[2].X<0.f ? box.MinEdge.Z:box.MaxEdge.Z);
                tmpBox.MaxEdge.Y = column[0].Y*(column[0].Y<0.f ? box.MinEdge.X:box.MaxEdge.X)+column[1].Y*(column[1].Y<0.f ? box.MinEdge.Y:box.MaxEdge.Y)+column[2].Y*(column[2].Y<0.f ? box.MinEdge.Z:box.MaxEdge.Z);
                tmpBox.MaxEdge.Z = column[0].Z*(column[0].Z<0.f ? box.MinEdge.X:box.MaxEdge.X)+column[1].Z*(column[1].Z<0.f ? box.MinEdge.Y:box.MaxEdge.Y)+column[2].Z*(column[2].Z<0.f ? box.MinEdge.Z:box.MaxEdge.Z);
                tmpBox.MinEdge += column[3];
                tmpBox.MaxEdge += column[3];

                box = tmpBox;
			}

			//! Calculates inverse of matrix. Slow.
			/** \return Returns false if there is no inverse matrix.*/
			inline bool makeInverse()
			{
                matrix4x3 temp ( EM4CONST_NOTHING );

                if (getInverse(temp))
                {
                    *this = temp;
                    return true;
                }

                return false;
			}


			//! Gets the inversed matrix of this one
			/** \param out: where result matrix is written to.
			\return Returns false if there is no inverse matrix. */
			inline bool getInverse(matrix4x3& out) const
			{
                /// Calculates the inverse of this Matrix
                /// The inverse is calculated using Cramers rule.
                /// If no inverse exists then 'false' is returned.
                float d = column[0].dotProduct(column[1].crossProduct(column[2]));

                if( core::iszero ( d, FLT_MIN ) )
                    return false;

                vectorSIMDf rows[3];
                rows[0].set(column[1].crossProduct(column[2]));
                rows[1].set(column[2].crossProduct(column[0]));
                rows[2].set(column[0].crossProduct(column[1]));

                rows[0] /= d;
                rows[1] /= d;
                rows[2] /= d;
                //transpose
                out.column[0].set(rows[0].X,rows[1].X,rows[2].X);
                out.column[1].set(rows[0].Y,rows[1].Y,rows[2].Y);
                out.column[2].set(rows[0].Z,rows[1].Z,rows[2].Z);

                //out.column[3] = out.column[0]*m.column[3].X+out.column[1]*m.column[3].Y+out.column[2]*m.column[3].Z;
                out.mulSub3x3With3x1(reinterpret_cast<float*>(out.column+3),reinterpret_cast<const float*>(column+3));
                out.column[3]  = -out.column[3];

                return true;
			}

			inline bool getSub3x3InverseTranspose(float* out) const
			{
                /// Calculates the inverse of this Matrix
                /// The inverse is calculated using Cramers rule.
                /// If no inverse exists then 'false' is returned.
                float d = column[0].dotProduct(column[1].crossProduct(column[2]));

                if( core::iszero ( d, FLT_MIN ) )
                    return false;

                vectorSIMDf rows[3];
                rows[0].set(column[1].crossProduct(column[2]));
                rows[1].set(column[2].crossProduct(column[0]));
                rows[2].set(column[0].crossProduct(column[1]));

                rows[0] /= d;
                rows[1] /= d;
                rows[2] /= d;

                out[0] = rows[0].pointer[0];
                out[1] = rows[0].pointer[1];
                out[2] = rows[0].pointer[2];
                out[3] = rows[1].pointer[0];
                out[4] = rows[1].pointer[1];
                out[5] = rows[1].pointer[2];
                out[6] = rows[2].pointer[0];
                out[7] = rows[2].pointer[1];
                out[8] = rows[2].pointer[2];

                return true;
			}


			//! Builds a left-handed look-at matrix.
			inline matrix4x3& buildCameraLookAtMatrixLH(
					const vector3df& position,
					const vector3df& target,
					const vector3df& upVector);

			//! Builds a right-handed look-at matrix.
			inline matrix4x3& buildCameraLookAtMatrixRH(
					const vector3df& position,
					const vector3df& target,
					const vector3df& upVector);


			//! Builds a matrix that rotates from one vector to another
			/** \param from: vector to rotate from
			\param to: vector to rotate to
			 */
			inline matrix4x3& buildRotateFromTo(const core::vector3df& from, const core::vector3df& to);

			//! Builds a combined matrix which translates to a center before rotation and translates from origin afterwards
			/** \param center Position to rotate around
			\param translate Translation applied after the rotation
			 */
			inline void setRotationCenter(const core::vector3df& center, const core::vector3df& translate);

			//! Builds a matrix which rotates a source vector to a look vector over an arbitrary axis
			/** \param camPos: viewer position in world coo
			\param center: object position in world-coo and rotation pivot
			\param translation: object final translation from center
			\param axis: axis to rotate about
			\param from: source vector to rotate from
			 */
			inline void buildAxisAlignedBillboard(const core::vector3df& camPos,
						const core::vector3df& center,
						const core::vector3df& translation,
						const core::vector3df& axis,
						const core::vector3df& from);


			//! Sets all matrix data members at once
			inline matrix4x3& setM(const float* data)
			{
			    memcpy(column,data,48);
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
            ret(0,i) = outColumn[i].X;
            ret(1,i) = outColumn[i].Y;
            ret(2,i) = outColumn[i].Z;
        }

        return ret;
    }

    inline matrix4 concatenateBFollowedByA(const matrix4& other_a,const matrix4x3& other_b )
    {
        vectorSIMDf inColumn[4];
        for (size_t i=0; i<4; i++)
            inColumn[i] = vectorSIMDf(other_a.pointer()+i*4);
        vectorSIMDf outColumn[4];

        outColumn[0] = inColumn[0]*other_b(0,0)+inColumn[1]*other_b(1,0)+inColumn[2]*other_b(2,0);
        outColumn[1] = inColumn[0]*other_b(0,1)+inColumn[1]*other_b(1,1)+inColumn[2]*other_b(2,1);
        outColumn[2] = inColumn[0]*other_b(0,2)+inColumn[1]*other_b(1,2)+inColumn[2]*other_b(2,2);
        outColumn[3] = inColumn[0]*other_b(0,3)+inColumn[1]*other_b(1,3)+inColumn[2]*other_b(2,3)+inColumn[3];

        return *reinterpret_cast<matrix4*>(outColumn);
    }

    inline matrix4 concatenatePreciselyBFollowedByA(const matrix4& other_a,const matrix4x3& other_b )
    {
        matrix4 ret;
        double aComponents[16];

        aComponents[ 0] = double(other_a(0,0))*double(other_b(0,0))+double(other_a(1,0))*double(other_b(1,0))+double(other_a(2,0))*double(other_b(2,0));
        aComponents[ 1] = double(other_a(0,1))*double(other_b(0,0))+double(other_a(1,1))*double(other_b(1,0))+double(other_a(2,1))*double(other_b(2,0));
        aComponents[ 2] = double(other_a(0,2))*double(other_b(0,0))+double(other_a(1,2))*double(other_b(1,0))+double(other_a(2,2))*double(other_b(2,0));
        aComponents[ 3] = double(other_a(0,3))*double(other_b(0,0))+double(other_a(1,3))*double(other_b(1,0))+double(other_a(2,3))*double(other_b(2,0));

        aComponents[ 4] = double(other_a(0,0))*double(other_b(0,1))+double(other_a(1,0))*double(other_b(1,1))+double(other_a(2,0))*double(other_b(2,1));
        aComponents[ 5] = double(other_a(0,1))*double(other_b(0,1))+double(other_a(1,1))*double(other_b(1,1))+double(other_a(2,1))*double(other_b(2,1));
        aComponents[ 6] = double(other_a(0,2))*double(other_b(0,1))+double(other_a(1,2))*double(other_b(1,1))+double(other_a(2,2))*double(other_b(2,1));
        aComponents[ 7] = double(other_a(0,3))*double(other_b(0,1))+double(other_a(1,3))*double(other_b(1,1))+double(other_a(2,3))*double(other_b(2,1));

        aComponents[ 8] = double(other_a(0,0))*double(other_b(0,2))+double(other_a(1,0))*double(other_b(1,2))+double(other_a(2,0))*double(other_b(2,2));
        aComponents[ 9] = double(other_a(0,1))*double(other_b(0,2))+double(other_a(1,1))*double(other_b(1,2))+double(other_a(2,1))*double(other_b(2,2));
        aComponents[10] = double(other_a(0,2))*double(other_b(0,2))+double(other_a(1,2))*double(other_b(1,2))+double(other_a(2,2))*double(other_b(2,2));
        aComponents[11] = double(other_a(0,3))*double(other_b(0,2))+double(other_a(1,3))*double(other_b(1,2))+double(other_a(2,3))*double(other_b(2,2));

        aComponents[12] = double(other_a(0,0))*double(other_b(0,3))+double(other_a(1,0))*double(other_b(1,3))+double(other_a(2,0))*double(other_b(2,3))+double(other_a(3,0));
        aComponents[13] = double(other_a(0,1))*double(other_b(0,3))+double(other_a(1,1))*double(other_b(1,3))+double(other_a(2,1))*double(other_b(2,3))+double(other_a(3,1));
        aComponents[14] = double(other_a(0,2))*double(other_b(0,3))+double(other_a(1,2))*double(other_b(1,3))+double(other_a(2,2))*double(other_b(2,3))+double(other_a(3,2));
        aComponents[15] = double(other_a(0,3))*double(other_b(0,3))+double(other_a(1,3))*double(other_b(1,3))+double(other_a(2,3))*double(other_b(2,3))+double(other_a(3,3));

        for (size_t j=0; j<4; j++)
        for (size_t i=0; i<4; i++)
            ret(j,i) = aComponents[j*4+i];

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

		const core::vector3d<float> invScale(core::reciprocal(scale.X),core::reciprocal(scale.Y),core::reciprocal(scale.Z));

		float nzd00 = mat(0,0)*invScale.X;
		float nzd11 = mat(1,1)*invScale.Y;
		float nzd22 = mat(2,2)*invScale.Z;
		//float trace = nzd00+nzd11+nzd22+1.f;

		float Y = -asinf(core::clamp(mat(2,0)*invScale.X, -1.f, 1.f));
		const float C = cosf(Y);
		Y *= RADTODEG64;

		float rotx, roty, X, Z;

		if (!core::iszero(C))
		{
			const float invC = core::reciprocal(C);
			rotx = nzd22 * invC;
			roty = mat(2,1) * invC * invScale.Y;
			X = atan2f( roty, rotx ) * RADTODEG64;
			rotx = nzd00 * invC;
			roty = mat(1,0) * invC * invScale.X;
			Z = atan2f( roty, rotx ) * RADTODEG64;
		}
		else
		{
			X = 0.0;
			roty = -mat(0,1) * invScale.Y;
			Z = atan2f( roty, nzd11 ) * RADTODEG64;
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
		const float t = 1.0 - c;

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


	// Builds a left-handed look-at matrix.
	inline matrix4x3& matrix4x3::buildCameraLookAtMatrixLH(
				const vector3df& position,
				const vector3df& target,
				const vector3df& upVector)
	{
		vector3df zaxis = target - position;
		vector3df xaxis = upVector.crossProduct(zaxis);

		float len = xaxis.X*xaxis.X+xaxis.Y*xaxis.Y+xaxis.Z*xaxis.Z;
		xaxis /= sqrtf(len);
		len = zaxis.X*zaxis.X+zaxis.Y*zaxis.Y+zaxis.Z*zaxis.Z;
		zaxis /= sqrtf(len);

		vector3df yaxis = zaxis.crossProduct(xaxis);

		column[0].X = (float)xaxis.X;
		column[0].Y = (float)yaxis.X;
		column[0].Z = (float)zaxis.X;

		column[1].X = (float)xaxis.Y;
		column[1].Y = (float)yaxis.Y;
		column[1].Z = (float)zaxis.Y;

		column[2].X = (float)xaxis.Z;
		column[2].Y = (float)yaxis.Z;
		column[2].Z = (float)zaxis.Z;

		column[3].X = (float)-xaxis.dotProduct(position);
		column[3].Y = (float)-yaxis.dotProduct(position);
		column[3].Z = (float)-zaxis.dotProduct(position);

		return *this;
	}


	// Builds a right-handed look-at matrix.
	inline matrix4x3& matrix4x3::buildCameraLookAtMatrixRH(
				const vector3df& position,
				const vector3df& target,
				const vector3df& upVector)
	{
		vector3df zaxis = position - target;
		vector3df xaxis = upVector.crossProduct(zaxis);

		float len = xaxis.X*xaxis.X+xaxis.Y*xaxis.Y+xaxis.Z*xaxis.Z;
		xaxis /= sqrtf(len);
		len = zaxis.X*zaxis.X+zaxis.Y*zaxis.Y+zaxis.Z*zaxis.Z;
		zaxis /= sqrtf(len);

		vector3df yaxis = zaxis.crossProduct(xaxis);

		column[0].X = (float)xaxis.X;
		column[0].Y = (float)yaxis.X;
		column[0].Z = (float)zaxis.X;

		column[1].X = (float)xaxis.Y;
		column[1].Y = (float)yaxis.Y;
		column[1].Z = (float)zaxis.Y;

		column[2].X = (float)xaxis.Z;
		column[2].Y = (float)yaxis.Z;
		column[2].Z = (float)zaxis.Z;

		column[3].X = (float)-xaxis.dotProduct(position);
		column[3].Y = (float)-yaxis.dotProduct(position);
		column[3].Z = (float)-zaxis.dotProduct(position);

		return *this;
	}


	// creates a new matrix as interpolated matrix from this and the passed one.
	inline matrix4x3 mix(const core::matrix4x3& a, const core::matrix4x3& b, const float& x)
	{
		matrix4x3 mat ( matrix4x3::EM4CONST_NOTHING );

		mat.getColumn(0) = a.getColumn(0)+(b.getColumn(0)-a.getColumn(0))*x;
		mat.getColumn(1) = a.getColumn(1)+(b.getColumn(1)-a.getColumn(1))*x;
		mat.getColumn(2) = a.getColumn(2)+(b.getColumn(2)-a.getColumn(2))*x;
		mat.getColumn(3) = a.getColumn(3)+(b.getColumn(3)-a.getColumn(3))*x;

		return mat;
	}


	//! Builds a matrix that rotates from one vector to another
	/** \param from: vector to rotate from
	\param to: vector to rotate to

		http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
	 */
	inline matrix4x3& matrix4x3::buildRotateFromTo(const core::vector3df& from, const core::vector3df& to)
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

		column[0].X = vt.X * v.X + ca;
		column[1].Y = vt.Y * v.Y + ca;
		column[2].Z = vt.Z * v.Z + ca;

		vt.X *= v.Y;
		vt.Z *= v.X;
		vt.Y *= v.Z;

		column[0].Y = vt.X - vs.Z;
		column[0].Z = vt.Z + vs.Y;

		column[1].X = vt.X + vs.Z;
		column[1].Z = vt.Y - vs.X;

		column[2].X = vt.Z - vs.Y;
		column[2].Y = vt.Y + vs.X;

		column[3].set(0.f,0.f,0.f);

		return *this;
	}

	//! Builds a matrix which rotates a source vector to a look vector over an arbitrary axis
	/** \param camPos: viewer position in world coord
	\param center: object position in world-coord, rotation pivot
	\param translation: object final translation from center
	\param axis: axis to rotate about
	\param from: source vector to rotate from
	 */
	inline void matrix4x3::buildAxisAlignedBillboard(
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
		const float ca = from.dotProduct(look);

		core::vector3df vt(up * (1.f - ca));

		column[0].X = static_cast<float>(vt.X * up.X + ca);
		column[1].Y = static_cast<float>(vt.Y * up.Y + ca);
		column[2].Z = static_cast<float>(vt.Z * up.Z + ca);

		vt.X *= up.Y;
		vt.Z *= up.X;
		vt.Y *= up.Z;

		column[0].Y = static_cast<float>(vt.X - vs.Z);
		column[0].Z = static_cast<float>(vt.Z + vs.Y);

		column[1].X = static_cast<float>(vt.X + vs.Z);
		column[1].Z = static_cast<float>(vt.Y - vs.X);

		column[2].X = static_cast<float>(vt.Z - vs.Y);
		column[2].Y = static_cast<float>(vt.Y + vs.X);

		setRotationCenter(center, translation);
	}


	//! Builds a combined matrix which translate to a center before rotation and translate afterwards
	inline void matrix4x3::setRotationCenter(const core::vector3df& center, const core::vector3df& translation)
	{
		column[3].X = -column[0].X*center.X - column[1].X*center.Y - column[2].X*center.Z + (center.X - translation.X );
		column[3].Y = -column[0].Y*center.X - column[1].Y*center.Y - column[2].Y*center.Z + (center.Y - translation.Y );
		column[3].Z = -column[0].Z*center.X - column[1].Z*center.Y - column[2].Z*center.Z + (center.Z - translation.Z );
	}


	//! global const identity matrix
	IRRLICHT_API extern const matrix4x3 IdentityMatrix;

} // end namespace core
} // end namespace irr

#endif

