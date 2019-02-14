//-----------------------------------------------------------------------------
// Copyright (c) 2012 GarageGames, LLC
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//-----------------------------------------------------------------------------

// Implementation taken from https://github.com/GarageGames/Torque3D and customized to Irrlicht

#ifndef __C_FORSYTH_VERTEX_CACHE_OPTIMIZER_H_INCLUDED__
#define __C_FORSYTH_VERTEX_CACHE_OPTIMIZER_H_INCLUDED__

#include <cstdint>
#include <cstring>
#include "irr/core/Types.h"

namespace irr { namespace asset
{

class IRR_FORCE_EBO CForsythVertexCacheOptimizer
{
	struct VertData
	{
		int32_t cachePosition;
		float score;
		uint32_t numReferences;
		uint32_t numUnaddedReferences;
		int32_t *triIndex;

		VertData() : cachePosition(-1), score(0.0f), numReferences(0), numUnaddedReferences(0), triIndex(NULL) {}
		~VertData() { if (triIndex != NULL) delete[] triIndex; triIndex = NULL; }
	};

	struct TriData
	{
		bool isInList;
		float score;
		uint32_t vertIdx[3];

		TriData() : isInList(false), score(0.0f) { memset(vertIdx, 0, sizeof(vertIdx)); }
	};

	class LRUCacheModel
	{
		struct LRUCacheEntry
		{
			LRUCacheEntry *next;
			uint32_t vIdx;
			VertData *vData;

			LRUCacheEntry() : next(NULL), vIdx(0), vData(NULL) {}
		};

		LRUCacheEntry *mCacheHead;

	public:
		LRUCacheModel() : mCacheHead(NULL) {}
		~LRUCacheModel();
		void enforceSize(const size_t maxSize, core::vector<uint32_t> &outTrisToUpdate);
		void useVertex(const uint32_t vIdx, VertData *vData);
		int32_t getCachePosition(const uint32_t vIdx);
	};


public:
	/**
	 This method will look at the index buffer for a triangle list, and generate
	 a new index buffer which is optimized using Tom Forsyth's paper:
	 "Linear-Speed Vertex Cache Optimization"
	 http://home.comcast.net/~tom_forsyth/papers/fast_vert_cache_opt.html
	 @param   numVerts Number of vertices indexed by the 'indices'
	 @param numIndices Number of elements in both 'indices' and 'outIndices'
	 @param    indices Input index buffer
	 @param outIndices Output index buffer

	 @note Both 'indices' and 'outIndices' can point to the same memory.*/
	template<typename IdxT> // IdxT is uint16_t or uint32_t
	void optimizeTriangleOrdering(const size_t _numVerts, const size_t _numIndices, const IdxT* _indices, IdxT* _outIndices) const;

private:
	static float score(const VertData &vertexData);
};

}}

#endif // __C_FORSYTH_VERTEX_CACHE_OPTIMIZER_H_INCLUDED__
