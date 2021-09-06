// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_OPEN_GL_QUERY_POOL_H_INCLUDED_
#define _NBL_VIDEO_C_OPEN_GL_QUERY_POOL_H_INCLUDED_

#include "nbl/core/SRange.h"
#include "nbl/core/decl/Types.h"
#include "nbl/video/IQueryPool.h"
#include "nbl/video/COpenGLCommon.h"
#include "nbl/video/IOpenGL_FunctionTable.h"

namespace nbl::video
{

class COpenGLQueryPool final : public IQueryPool
{
	protected:
		virtual ~COpenGLQueryPool();

		core::vector<GLuint> queries;

	public:
		COpenGLQueryPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, IOpenGL_FunctionTable* gl, IQueryPool::SCreationParams&& _params) 
			: IQueryPool(std::move(dev), std::move(_params))
		{

			if(_params.queryType == EQT_OCCLUSION)
			{
				queries.resize(params.queryCount);
				gl->extGlCreateQueries(GL_SAMPLES_PASSED, _params.queryCount, queries.data());
			}
			else if(_params.queryType == EQT_TIMESTAMP)
			{
				queries.resize(params.queryCount);
				gl->extGlCreateQueries(GL_TIMESTAMP, _params.queryCount, queries.data());
			}
			else if(_params.queryType == EQT_TRANSFORM_FEEDBACK_STREAM_EXT)
			{
				// Vulkan Transform feedback queries write two integers;
				// The first integer is the number of primitives successfully written to the corresponding transform feedback buffer
				// and the second is the number of primitives output to the vertex stream.
				// But in OpenGL there you need twice the queries to get both values.
				queries.resize(params.queryCount * 2);
				gl->extGlCreateQueries(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, _params.queryCount, queries.data());
				gl->extGlCreateQueries(GL_PRIMITIVES_GENERATED, _params.queryCount, queries.data() + params.queryCount);
				_NBL_TODO();
			}
			else
			{
				assert(false && "QueryType is not supported.");
			}
		}

		inline core::SRange<const GLuint> getQueries() const
		{
			return core::SRange<const GLuint>(queries.data(), queries.data() + queries.size());
		}
};

} // end namespace nbl::video


#endif

