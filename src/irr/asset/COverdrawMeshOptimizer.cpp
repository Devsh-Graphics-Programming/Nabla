#include "COverdrawMeshOptimizer.h"

#include <vector>
#include <functional>

#include "CMeshManipulator.h"
#include "os.h"

namespace irr { namespace asset
{

asset::ICPUMeshBuffer* COverdrawMeshOptimizer::createOptimized(asset::ICPUMeshBuffer* _inbuffer, bool _createNew, float _threshold)
{
	if (!_inbuffer)
		return NULL;

	const asset::E_INDEX_TYPE indexType = _inbuffer->getIndexType();
	if (indexType == asset::EIT_UNKNOWN)
		return NULL;

	const size_t indexSize = indexType == asset::EIT_16BIT ? 2u : 4u;

	asset::ICPUMeshBuffer* outbuffer = _createNew ? CMeshManipulator().createMeshBufferDuplicate(_inbuffer) : _inbuffer;

	void* const indices = outbuffer->getIndices();
	if (!indices)
	{
#ifdef _IRR_DEBUG
		os::Printer::log("Overdraw optimization: no index buffer -- mesh buffer left unchanged.");
#endif
		return outbuffer;
	}

	const size_t idxCount = outbuffer->getIndexCount();
	void* indicesCopy = _IRR_ALIGNED_MALLOC(indexSize*idxCount,_IRR_SIMD_ALIGNMENT);
	memcpy(indicesCopy, indices, indexSize*idxCount);
	const size_t vertexCount = outbuffer->calcVertexCount();
	core::vector<core::vectorSIMDf> vertexPositions;
	{
		core::vectorSIMDf pos;
		for (size_t i = 0; i < vertexCount; ++i)
		{
			outbuffer->getAttribute(pos, outbuffer->getPositionAttributeIx(), i);
			vertexPositions.push_back(pos);
		}
	}

	uint32_t* const hardClusters = (uint32_t*)_IRR_ALIGNED_MALLOC((idxCount/3) * 4,_IRR_SIMD_ALIGNMENT);
	const size_t hardClusterCount = indexType == asset::EIT_16BIT ?
		genHardBoundaries(hardClusters, (uint16_t*)indices, idxCount, vertexCount) :
		genHardBoundaries(hardClusters, (uint32_t*)indices, idxCount, vertexCount);

	uint32_t* const softClusters = (uint32_t*)_IRR_ALIGNED_MALLOC((idxCount/3 + 1) * 4,_IRR_SIMD_ALIGNMENT);
	const size_t softClusterCount = indexType == asset::EIT_16BIT ?
		genSoftBoundaries(softClusters, (uint16_t*)indices, idxCount, vertexCount, hardClusters, hardClusterCount, _threshold) :
		genSoftBoundaries(softClusters, (uint32_t*)indices, idxCount, vertexCount, hardClusters, hardClusterCount, _threshold);

	ClusterSortData* sortedData = (ClusterSortData*)_IRR_ALIGNED_MALLOC(softClusterCount*sizeof(ClusterSortData),_IRR_SIMD_ALIGNMENT);
	if (indexType == asset::EIT_16BIT)
		calcSortData(sortedData, (uint16_t*)indices, idxCount, vertexPositions, softClusters, softClusterCount);
	else
		calcSortData(sortedData, (uint32_t*)indices, idxCount, vertexPositions, softClusters, softClusterCount);

	std::sort(sortedData, sortedData + softClusterCount, std::greater<ClusterSortData>());

	for (size_t it = 0, jt = 0; it < softClusterCount; ++it)
	{
		const uint32_t cluster = sortedData[it].cluster;

		size_t start = softClusters[cluster];
		size_t end = (cluster+1 < softClusterCount) ? softClusters[cluster+1] : idxCount/3;

		if (indexType == asset::EIT_16BIT)
		{
			for (size_t i = start; i < end; ++i)
			{
				((uint16_t*)indices)[jt++] = ((uint16_t*)indicesCopy)[3*i + 0];
				((uint16_t*)indices)[jt++] = ((uint16_t*)indicesCopy)[3*i + 1];
				((uint16_t*)indices)[jt++] = ((uint16_t*)indicesCopy)[3*i + 2];
			}
		}
		else
		{
			for (size_t i = start; i < end; ++i)
			{
				((uint32_t*)indices)[jt++] = ((uint32_t*)indicesCopy)[3*i + 0];
				((uint32_t*)indices)[jt++] = ((uint32_t*)indicesCopy)[3*i + 1];
				((uint32_t*)indices)[jt++] = ((uint32_t*)indicesCopy)[3*i + 2];
			}
		}
	}

	_IRR_ALIGNED_FREE(indicesCopy);
	_IRR_ALIGNED_FREE(hardClusters);
	_IRR_ALIGNED_FREE(softClusters);
	_IRR_ALIGNED_FREE(sortedData);

	return outbuffer;
}

template<typename IdxT>
size_t COverdrawMeshOptimizer::genHardBoundaries(uint32_t* _dst, const IdxT* _indices, size_t _idxCount, size_t _vtxCount)
{
	size_t* cacheTimestamps = (size_t*)_IRR_ALIGNED_MALLOC(sizeof(size_t)*_vtxCount,_IRR_SIMD_ALIGNMENT);
	memset(cacheTimestamps, 0, sizeof(size_t)*_vtxCount);

	size_t timestamp = CACHE_SIZE + 1;

	const size_t faceCount = _idxCount / 3;

	size_t retval = 0u;
	for (size_t i = 0u; i < faceCount; ++i)
	{
		size_t misses = updateCache(_indices[3*i + 0], _indices[3*i + 1], _indices[3*i + 2], cacheTimestamps, timestamp);

		// when all three vertices are not in the cache it's usually relatively safe to assume that this is a new patch in the mesh
		// that is disjoint from previous vertices; sometimes it might come back to reference existing vertices but that frequently
		// suggests an inefficiency in the vertex cache optimization algorithm
		// usually the first triangle has 3 misses unless it's degenerate - thus we make sure the first cluster always starts with 0
		if (!i || misses == 3)
			_dst[retval++] = (uint32_t)i;
	}

	_IRR_ALIGNED_FREE(cacheTimestamps);

	return retval;
}

template<typename IdxT>
size_t COverdrawMeshOptimizer::genSoftBoundaries(uint32_t* _dst, const IdxT* _indices, size_t _idxCount, size_t _vtxCount, const uint32_t* _clusters, size_t _clusterCount, float _threshold)
{
	size_t* cacheTimestamps = (size_t*)_IRR_ALIGNED_MALLOC(sizeof(size_t)*_vtxCount,_IRR_SIMD_ALIGNMENT);
	memset(cacheTimestamps, 0, sizeof(size_t)*_vtxCount);

	size_t timestamp = 0u;

	size_t retval = 0u;

	for (size_t i = 0u; i < _clusterCount; ++i)
	{
		const size_t start = _clusters[i];
		const size_t end = (i + 1 < _clusterCount) ? _clusters[i+1] : _idxCount/3;

		_IRR_DEBUG_BREAK_IF(start > end);

		timestamp += CACHE_SIZE + 1; // reset cache

		size_t clusterMisses = 0u; // cluster ACMR
		for (size_t j = start; j < end; ++j)
			clusterMisses += updateCache(_indices[j*3 + 0], _indices[j*3 + 1], _indices[j*3 + 2], cacheTimestamps, timestamp);

		const float clusterThreshold = _threshold * (float(clusterMisses) / (end - start));

		_dst[retval++] = (uint32_t)start;

		timestamp += CACHE_SIZE + 1; // reset cache

		size_t runningMisses = 0u;
		size_t runningFaces = 0u;
		for (size_t j = start; j < end; ++j)
		{
			runningMisses += updateCache(_indices[j*3 + 0], _indices[j*3 + 1], _indices[j*3 + 2], cacheTimestamps, timestamp);
			++runningFaces;

			if (float(runningMisses)/runningFaces <= clusterThreshold)
			{
				// we have reached the target ACMR with the current triangle so we need to start a new cluster on the next one
				// note that this may mean that we add 'end` to destination for the last triangle, which will imply that the last
				// cluster is empty; however, the 'pop back' after the loop will clean it up
				_dst[retval++] = (uint32_t)j + 1;

				timestamp += CACHE_SIZE + 1; // reset cache

				runningMisses = 0u;
				runningFaces = 0u;
			}
		}

		// each time we reach the target ACMR we flush the cluster
		// this means that the last cluster is by definition not very good - there are frequent cases where we are left with a few triangles
		// in the last cluster, producing a very bad ACMR and significantly penalizing the overall results
		// thus we remove the last cluster boundary, merging the last complete cluster with the last incomplete one
		// there are sometimes cases when the last cluster is actually good enough - in which case the code above would have added 'end'
		// to the cluster boundary array which we need to remove anyway - this code will do that automatically
		if (_dst[retval - 1] != start)
			--retval;
	}

	_IRR_ALIGNED_FREE(cacheTimestamps);

	_IRR_DEBUG_BREAK_IF(retval < _clusterCount || retval > _idxCount/3u)

	return retval;
}

template<typename IdxT>
void COverdrawMeshOptimizer::calcSortData(ClusterSortData* _dst, const IdxT* _indices, size_t _idxCount, const core::vector<core::vectorSIMDf>& _positions, const uint32_t* _clusters, size_t _clusterCount)
{
	core::vectorSIMDf meshCentroid;
	for (size_t i = 0u; i < _idxCount; ++i)
		meshCentroid += _positions[_indices[i]];
	meshCentroid /= (float)_idxCount;

	for (size_t cluster = 0; cluster < _clusterCount; ++cluster)
	{
		const size_t begin = _clusters[cluster] * 3;
		const size_t end = (_clusterCount > cluster + 1) ? _clusters[cluster+1] * 3 : _idxCount;
		_IRR_DEBUG_BREAK_IF(begin > end);

		float clusterArea = 0.f;
		core::vectorSIMDf clusterCentroid;
		core::vectorSIMDf clusterNormal;

		for (size_t i = begin; i < end; i += 3)
		{
			const core::vectorSIMDf& p0 = _positions[_indices[i+0]];
			const core::vectorSIMDf& p1 = _positions[_indices[i+1]];
			const core::vectorSIMDf& p2 = _positions[_indices[i+2]];

			const core::vectorSIMDf normal = (p1 - p0).crossProduct(p2 - p0);

			const float area = normal.getLengthAsFloat();

			clusterCentroid += (p0 + p1 + p2) * (area / 3.f);
			clusterNormal += normal;
			clusterArea += area;
		}

		const float invClusterArea = !clusterArea ? 0.f : 1.f/clusterArea;

		clusterCentroid *= invClusterArea;
		clusterNormal = core::normalize(clusterNormal);

		core::vectorSIMDf centroidVec = clusterCentroid - meshCentroid;

		_dst[cluster].cluster = (uint32_t)cluster;
		_dst[cluster].dot = centroidVec.dotProduct(clusterNormal).x;
	}
}

size_t COverdrawMeshOptimizer::updateCache(uint32_t _a, uint32_t _b, uint32_t _c, size_t* _cacheTimestamps, size_t& _timestamp)
{
	size_t cacheMisses = 0u;

	if (_timestamp - _cacheTimestamps[_a] > CACHE_SIZE)
	{
		_cacheTimestamps[_a] = _timestamp++;
		++cacheMisses;
	}
	if (_timestamp - _cacheTimestamps[_b] > CACHE_SIZE)
	{
		_cacheTimestamps[_b] = _timestamp++;
		++cacheMisses;
	}
	if (_timestamp - _cacheTimestamps[_c] > CACHE_SIZE)
	{
		_cacheTimestamps[_c] = _timestamp++;
		++cacheMisses;
	}

	return cacheMisses;
}

}} // irr::scene
