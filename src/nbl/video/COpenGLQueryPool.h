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
				queries.resize(_params.queryCount);
				gl->extGlCreateQueries(GL_SAMPLES_PASSED, _params.queryCount, queries.data());
			}
			else if(_params.queryType == EQT_TIMESTAMP)
			{
				queries.resize(_params.queryCount);
				gl->extGlCreateQueries(GL_TIMESTAMP, _params.queryCount, queries.data());
			}
			else
			{
				// TODO: Add ARB_pipeline_statistics support: https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_pipeline_statistics_query.txt
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
				assert(false);
				return 0u; // is 0 an invalid GLuint?
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
				else
				{
					assert(false && "QueryType is not supported.");
				}
			}
		}

		inline bool resetQueries(IOpenGL_FunctionTable* gl, uint32_t query, uint32_t queryCount)
		{
			// NOTE: There is no Reset Queries on OpenGL but to make the queries invalid/unavailable and not return the previous ones we just delete the queries and recreate them.
			// TODO: Needs test
			size_t querySize = queries.size();

			if(query + queryCount > querySize)
			{
				assert(false);
				return false;
			}

			if(params.queryType == EQT_OCCLUSION)
			{
				gl->glQuery.pglDeleteQueries(queryCount, queries.data() + query);
				gl->extGlCreateQueries(GL_SAMPLES_PASSED, queryCount, queries.data() + query);
			}
			else if(params.queryType == EQT_TIMESTAMP)
			{
				gl->glQuery.pglDeleteQueries(queryCount, queries.data() + query);
				gl->extGlCreateQueries(GL_TIMESTAMP, queryCount, queries.data() + query);
			}

			return true;
		}

};

} // end namespace nbl::video


#endif

