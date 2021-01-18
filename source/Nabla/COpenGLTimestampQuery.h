// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_OPENGL_TIMESTAMP_QUERY_H_INCLUDED__
#define __NBL_C_OPENGL_TIMESTAMP_QUERY_H_INCLUDED__

#include "IGPUTimestampQuery.h"
#ifdef _NBL_COMPILE_WITH_OPENGL_
#include "COpenGLExtensionHandler.h"

namespace nbl
{
namespace video
{


class COpenGLTimestampQuery : public IGPUTimestampQuery
{
    protected:
        virtual ~COpenGLTimestampQuery()
        {
            COpenGLExtensionHandler::extGlDeleteQueries(1,&object);
        }

    public:
        COpenGLTimestampQuery() : object(0), ready(false), hasResult(false), cachedCompletedTime(0xdeadbeefbadc0ffeu)
        {
            COpenGLExtensionHandler::extGlCreateQueries(GL_TIMESTAMP,1,&object);
            COpenGLExtensionHandler::extGlQueryCounter(object,GL_TIMESTAMP);
        }

		virtual bool isQueryReady()
		{
		    if (ready)
                return true;

            GLuint available = GL_FALSE;
            COpenGLExtensionHandler::extGlGetQueryObjectuiv(object,GL_QUERY_RESULT_AVAILABLE,&available);
            ready = available!=GL_FALSE;

            return ready;
		}

        virtual uint64_t getTimestampWhenCompleted()
        {
		    if (hasResult)
                return cachedCompletedTime;

            COpenGLExtensionHandler::extGlGetQueryObjectui64v(object,GL_QUERY_RESULT,&cachedCompletedTime);
            return cachedCompletedTime;
        }
    private:
        GLuint object;

        bool ready;
        bool hasResult;
        uint64_t cachedCompletedTime;
};

}
}
#endif // _NBL_COMPILE_WITH_OPENGL_

#endif


