// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_OPEN_GL_QUERY_POOL_H_INCLUDED_
#define _NBL_VIDEO_C_OPEN_GL_QUERY_POOL_H_INCLUDED_

#include "nbl/video/IQueryPool.h"
#include "nbl/video/COpenGLCommon.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
namespace nbl::video
{

class COpenGLQueryPool final : public IQueryPool
{
	protected:
		virtual ~COpenGLQueryPool();

	public:
		COpenGLQueryPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, IOpenGL_FunctionTable* gl, IQueryPool::SCreationParams&& _params) 
			: IQueryPool(std::move(dev), std::move(_params))
		{

		}


};

} // end namespace nbl::video


#endif

