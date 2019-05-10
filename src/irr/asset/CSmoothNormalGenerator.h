#ifndef __C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED__
#define __C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED__

#include <iostream>
#include "irr/asset/ICPUMeshBuffer.h"
#include "irr/core/math/irrMath.h"

namespace irr 
{	
namespace asset 
{


class CSmoothNormalGenerator
{
public:
	static asset::ICPUMeshBuffer* calculateNormals(asset::ICPUMeshBuffer* buffer, float creaseAngle);

private:
	struct Vertex
	{
		uint32_t indexOffset;
		core::vector3df_SIMD normal;
		core::vector3df_SIMD parentTriangleFaceNormal;
		float wage;
	};

private:
	static core::vector<Vertex> setupData(asset::ICPUMeshBuffer* buffer, float creaseAngle);
	static void processConnectedVertices(asset::ICPUMeshBuffer* buffer, core::vector<Vertex>& vertices, float creaseAngle);

	CSmoothNormalGenerator() {};
	~CSmoothNormalGenerator() {};
};



}
}

#endif