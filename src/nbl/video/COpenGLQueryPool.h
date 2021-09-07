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

		// queries.size() is a multiple of params.queryCount
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

		inline GLuint getQueryAt(uint32_t index) const
		{
			if(index < queries.size())
			{
				return queries[index];
			}
			else
			{
				return 0; // is 0 an invalid GLuint?
			}
		}

		inline void beginQuery(IOpenGL_FunctionTable* gl, uint32_t queryIndex, E_QUERY_CONTROL_FLAGS flags) const
		{
			if(gl != nullptr)
			{
				if(params.queryType == EQT_OCCLUSION)
				{
					GLuint query = getQueryAt(queryIndex);
					gl->glQuery.pglBeginQuery(GL_SAMPLES_PASSED, query);
				}
				else if(params.queryType == EQT_TIMESTAMP)
				{
					assert(false && "TIMESTAMP Query doesn't work with begin/end functions.");
				}
				else if(params.queryType == EQT_TRANSFORM_FEEDBACK_STREAM_EXT)
				{
					GLuint query1 = getQueryAt(queryIndex);
					GLuint query2 = getQueryAt(queryIndex + params.queryCount);
					gl->glQuery.pglBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query1);
					gl->glQuery.pglBeginQuery(GL_PRIMITIVES_GENERATED, query2);
				}
				else
				{
					assert(false && "QueryType is not supported.");
				}
			}
		}
		
		inline void endQuery(IOpenGL_FunctionTable* gl, uint32_t queryIndex) const
		{
			// End Function doesn't use queryIndex
			if(gl != nullptr)
			{
				if(params.queryType == EQT_OCCLUSION)
				{
					gl->glQuery.pglEndQuery(GL_SAMPLES_PASSED);
				}
				else if(params.queryType == EQT_TIMESTAMP)
				{
					assert(false && "TIMESTAMP Query doesn't work with begin/end functions.");
				}
				else if(params.queryType == EQT_TRANSFORM_FEEDBACK_STREAM_EXT)
				{
					gl->glQuery.pglEndQuery(GL_PRIMITIVES_GENERATED);
					gl->glQuery.pglEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
				}
				else
				{
					assert(false && "QueryType is not supported.");
				}
			}
		}
		
		inline void beginQueryIndexed(IOpenGL_FunctionTable* gl, uint32_t queryIndex, uint32_t index, E_QUERY_CONTROL_FLAGS flags) const
		{
			if(gl != nullptr)
			{
				if(params.queryType == EQT_OCCLUSION)
				{
					// if(index != 0)
					// 	assert(false && "OCCLUSION Query doesn't work with begin/end Indexed functions.");
					GLuint query = getQueryAt(queryIndex);
					gl->glQuery.pglBeginQueryIndexed(GL_SAMPLES_PASSED, index, query);
				}
				else if(params.queryType == EQT_TIMESTAMP)
				{
					assert(false && "TIMESTAMP Query doesn't work with begin/end functions.");
				}
				else if(params.queryType == EQT_TRANSFORM_FEEDBACK_STREAM_EXT)
				{
					GLuint query1 = getQueryAt(queryIndex);
					GLuint query2 = getQueryAt(queryIndex + params.queryCount);
					gl->glQuery.pglBeginQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, index, query1);
					gl->glQuery.pglBeginQueryIndexed(GL_PRIMITIVES_GENERATED, index, query2);
				}
				else
				{
					assert(false && "QueryType is not supported.");
				}
			}
		}
		
		inline void endQueryIndexed(IOpenGL_FunctionTable* gl, uint32_t queryIndex, uint32_t index) const
		{
			// End Function doesn't use queryIndex
			if(gl != nullptr)
			{
				if(params.queryType == EQT_OCCLUSION)
				{
					gl->glQuery.pglEndQueryIndexed(GL_SAMPLES_PASSED, index);
				}
				else if(params.queryType == EQT_TIMESTAMP)
				{
					assert(false && "TIMESTAMP Query doesn't work with begin/end functions.");
				}
				else if(params.queryType == EQT_TRANSFORM_FEEDBACK_STREAM_EXT)
				{
					gl->glQuery.pglEndQueryIndexed(GL_PRIMITIVES_GENERATED, index);
					gl->glQuery.pglEndQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, index);
				}
				else
				{
					assert(false && "QueryType is not supported.");
				}
			}
		}
};

} // end namespace nbl::video


#endif

