// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_AABBOX_3D_H_INCLUDED__
#define __NBL_AABBOX_3D_H_INCLUDED__

/**
 * @TODO: This seems to be used for only one function here which is not invoked anywhere in the
 *   current state of the codebase and line3d is effectively replaced by CDraw3DLine extension
 *   two suggestions:
 *   - can inherit from line3d for all CDraw3DLine debug extension and cover all funcs from the superclass?
 *   - can remove all line3d instances/usages and unify all line drawing funcs to CDraw3DLine?
 *     (not the impl yet though to complete CDraw3DLine if smth missing,
 *     and use not only for debug draw, but any other line3d draw)
 */
#include "line3d.h"

namespace nbl
{
namespace core
{

//! Axis aligned bounding box in 3d dimensional space.
/** Has some useful methods used with occlusion culling or clipping.
 * @TODO UVector was introduced to allow for usage other than of vector3d<float> (i.e. vectorSIMDf)
 *   and defaulted to allow for gradual migration of vector3d-based usage/implementation of aabbox3df
 *   to vectorSIMDf (i.e. vector3df_SIMD) everywhere in the codebase
*/
template<class T, class UVector>
class aabbox3d // : public AllocationOverrideDefault ?
{
  static_assert(
    std::is_convertible<vector3d<T>, UVector>::value ||
    std::is_convertible<vectorSIMDf, UVector>::value ,
    "UVector should be of either vector3d<T>, vectorSIMDf or their typedefs"
  );

	public:

		//! Default Constructor.
		aabbox3d()
    : MinEdge(-1,-1,-1), MaxEdge(1,1,1) {}
		//! Constructor with min edge and max edge.
		aabbox3d(const UVector& min, const UVector& max)
    : MinEdge(min), MaxEdge(max) {}
		//! Constructor with min edge and max edge as single values, not vectors.
		aabbox3d(T minx, T miny, T minz, T maxx, T maxy, T maxz)
    : MinEdge(minx, miny, minz), MaxEdge(maxx, maxy, maxz) {}
    //! Constructor with only one point.
    explicit aabbox3d(const UVector& _init)
    : MinEdge(_init), MaxEdge(_init) {}

		// operators
		//! Equality operator
		/** \param other box to compare with.
		\return True if both boxes are equal, else false. */
		inline bool operator==(const aabbox3d<T>& other) const { return !operator!=(other); }
		//! Inequality operator
		/** \param other box to compare with.
		\return True if both boxes are different, else false. */
		inline bool operator!=(const aabbox3d<T>& other) const { return MinEdge!=other.MinEdge || other.MaxEdge!=MaxEdge; }

		// functions

		//! Resets the bounding box to a one-point box.
		/** \param x X coord of the point.
		\param y Y coord of the point.
		\param z Z coord of the point. */
		void reset(T x, T y, T z)
		{
			MaxEdge.set(x,y,z);
			MinEdge = MaxEdge;
		}

		//! Resets the bounding box.
		/** \param initValue New box to set this one to. */
		void reset(const aabbox3d<T>& initValue)
		{
			*this = initValue;
		}

		//! Resets the bounding box to a one-point box.
		/** \param initValue New point. */
		void reset(const UVector& initValue)
		{
			MaxEdge = initValue;
			MinEdge = initValue;
		}

		//! Adds another bounding box
		/** The box grows bigger, if the new box was outside of the box.
		\param b: Other bounding box to add into this box. */
		template<typename TBox = aabbox3d<T, UVector>>
		void addInternalBox(const TBox& b)
		{
      static_assert(
        std::is_base_of<aabbox3d<T, UVector>, TBox>::value,
        "TBox should derive from aabbox3d<T, UVector>"
      );

			addInternalPoint(b.MaxEdge);
			addInternalPoint(b.MinEdge);
		}

    //! Adds a point to the bounding box
    /** The box grows bigger, if point was outside of the box.
    \param p: Point to add into the box. */
    void addInternalPoint(const UVector& p)
    {
      addInternalPoint(p.X, p.Y, p.Z);
    }

		//! Adds a point to the bounding box
		/** The box grows bigger, if point is outside of the box.
		\param x X coordinate of the point to add to this box.
		\param y Y coordinate of the point to add to this box.
		\param z Z coordinate of the point to add to this box. */
		void addInternalPoint(T x, T y, T z)
		{
			if (x>MaxEdge.X) MaxEdge.X = x;
			if (y>MaxEdge.Y) MaxEdge.Y = y;
			if (z>MaxEdge.Z) MaxEdge.Z = z;

			if (x<MinEdge.X) MinEdge.X = x;
			if (y<MinEdge.Y) MinEdge.Y = y;
			if (z<MinEdge.Z) MinEdge.Z = z;
		}

		//! Get center of the bounding box
		/** \return Center of the bounding box. */
		UVector getCenter() const
		{
			return (MinEdge + MaxEdge) / 2;
		}

		//! Get extent of the box (maximal distance of two points in the box)
		/** \return Extent of the bounding box. */
		UVector getExtent() const
		{
			return MaxEdge - MinEdge;
		}

		//! Check if the box is empty.
		/** This means that there is no space between the min and max edge.
		\return True if box is empty, else false. */
		bool isEmpty() const
		{
			return MinEdge.equals ( MaxEdge );
		}

		//! Get the volume enclosed by the box in cubed units
		T getVolume() const
		{
			const UVector e = getExtent();
			return e.X * e.Y * e.Z;
		}

		//! Get the surface area of the box in squared units
		T getArea() const
		{
			const UVector e = getExtent();
			return 2*(e.X*e.Y + e.X*e.Z + e.Y*e.Z);
		}

		//! Stores all 8 edges of the box into an array
		/** \param edges: Pointer to array of 8 edges. */
		template<class vectorT>
		void getEdges(vectorT* edges) const
		{
			const UVector middle = getCenter();
			const UVector diag = middle - MaxEdge;

			/*
			Edges are stored in this way:
			Hey, am I an ascii artist, or what? :) niko.
                   /3--------/7
                  / |       / |
                 /  |      /  |
                1---------5   |
                |  /2- - -|- -6
                | /       |  /
                |/        | /
                0---------4/
			*/

			edges[0].set(middle.X + diag.X, middle.Y + diag.Y, middle.Z + diag.Z);
			edges[1].set(middle.X + diag.X, middle.Y - diag.Y, middle.Z + diag.Z);
			edges[2].set(middle.X + diag.X, middle.Y + diag.Y, middle.Z - diag.Z);
			edges[3].set(middle.X + diag.X, middle.Y - diag.Y, middle.Z - diag.Z);
			edges[4].set(middle.X - diag.X, middle.Y + diag.Y, middle.Z + diag.Z);
			edges[5].set(middle.X - diag.X, middle.Y - diag.Y, middle.Z + diag.Z);
			edges[6].set(middle.X - diag.X, middle.Y + diag.Y, middle.Z - diag.Z);
			edges[7].set(middle.X - diag.X, middle.Y - diag.Y, middle.Z - diag.Z);

      // TODO: remove once done testing
      edges[8].set(MinEdge.X, MinEdge.Y, MinEdge.Z); // red
      edges[9].set(MaxEdge.X, MaxEdge.Y, MaxEdge.Z); // blue
    }

		//! Repairs the box.
		/** Necessary if for example MinEdge and MaxEdge are swapped. */
		void repair()
		{
			T t;

			if (MinEdge.X > MaxEdge.X)
				{ t=MinEdge.X; MinEdge.X = MaxEdge.X; MaxEdge.X=t; }
			if (MinEdge.Y > MaxEdge.Y)
				{ t=MinEdge.Y; MinEdge.Y = MaxEdge.Y; MaxEdge.Y=t; }
			if (MinEdge.Z > MaxEdge.Z)
				{ t=MinEdge.Z; MinEdge.Z = MaxEdge.Z; MaxEdge.Z=t; }
		}

		//! Determines if a point is within this box.
		/** Border is included (IS part of the box)!
		\param p: Point to check.
		\return True if the point is within the box and false if not */
		bool isPointInside(const UVector& p) const
		{
			return (p.X >= MinEdge.X && p.X <= MaxEdge.X &&
				p.Y >= MinEdge.Y && p.Y <= MaxEdge.Y &&
				p.Z >= MinEdge.Z && p.Z <= MaxEdge.Z);
		}

		//! Determines if a point is within this box and not its borders.
		/** Border is excluded (NOT part of the box)!
		\param p: Point to check.
		\return True if the point is within the box and false if not. */
		bool isPointTotalInside(const UVector& p) const
		{
			return (p.X > MinEdge.X && p.X < MaxEdge.X &&
				p.Y > MinEdge.Y && p.Y < MaxEdge.Y &&
				p.Z > MinEdge.Z && p.Z < MaxEdge.Z);
		}

		//! Check if this box is completely inside the 'other' box.
		/** \param other: Other box to check against.
		\return True if this box is completly inside the other box,
		otherwise false. */
		bool isFullInside(const aabbox3d<T>& other) const
		{
			return (MinEdge.X >= other.MinEdge.X && MinEdge.Y >= other.MinEdge.Y && MinEdge.Z >= other.MinEdge.Z &&
				MaxEdge.X <= other.MaxEdge.X && MaxEdge.Y <= other.MaxEdge.Y && MaxEdge.Z <= other.MaxEdge.Z);
		}

		//! Determines if the axis-aligned box intersects with another axis-aligned box.
		/** \param other: Other box to check a intersection with.
		\return True if there is an intersection with the other box,
		otherwise false. */
		bool intersectsWithBox(const aabbox3d<T>& other) const
		{
			return (MinEdge.X <= other.MaxEdge.X && MinEdge.Y <= other.MaxEdge.Y && MinEdge.Z <= other.MaxEdge.Z &&
				MaxEdge.X >= other.MinEdge.X && MaxEdge.Y >= other.MinEdge.Y && MaxEdge.Z >= other.MinEdge.Z);
		}

		//! Tests if the box intersects with a line
		/** \param line: Line to test intersection with.
		\return True if there is an intersection , else false. */
		bool intersectsWithLine(const line3d<T>& line) const
		{
			return intersectsWithLine(line.getMiddle(), line.getVector().normalize(),
					(T)(line.getLength() * 0.5));
		}

		//! Tests if the box intersects with a line
		/** \param linemiddle Center of the line.
		\param linevect Vector of the line.
		\param halflength Half length of the line.
		\return True if there is an intersection, else false. */
		bool intersectsWithLine(const UVector& linemiddle,
					const UVector& linevect, T halflength) const
		{
			const UVector e = getExtent() * (T)0.5;
			const UVector t = getCenter() - linemiddle;

			if ((fabs(t.X) > e.X + halflength * fabs(linevect.X)) ||
				(fabs(t.Y) > e.Y + halflength * fabs(linevect.Y)) ||
				(fabs(t.Z) > e.Z + halflength * fabs(linevect.Z)) )
				return false;

			T r = e.Y * (T)fabs(linevect.Z) + e.Z * (T)fabs(linevect.Y);
			if (fabs(t.Y*linevect.Z - t.Z*linevect.Y) > r )
				return false;

			r = e.X * (T)fabs(linevect.Z) + e.Z * (T)fabs(linevect.X);
			if (fabs(t.Z*linevect.X - t.X*linevect.Z) > r )
				return false;

			r = e.X * (T)fabs(linevect.Y) + e.Y * (T)fabs(linevect.X);
			if (fabs(t.X*linevect.Y - t.Y*linevect.X) > r)
				return false;

			return true;
		}

		//! The near edge
		UVector MinEdge;

		//! The far edge
		UVector MaxEdge;

    float Area = 0;   // quality measure of the bounding box
};

//! Typedef for a SIMD float 3d bounding box.
typedef aabbox3d<float, vectorSIMDf> aabbox3dsf;
//! Typedef for a float 3d bounding box.
typedef aabbox3d<float, vector3d<float>> aabbox3df;
//! Typedef for an integer 3d bounding box.
typedef aabbox3d<int32_t, vector3d<int32_t>> aabbox3di;

} // end namespace core
} // end namespace nbl

#endif

