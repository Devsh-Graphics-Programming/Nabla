#include "COpenGLOcclusionQuery.h"

namespace irr
{
namespace video
{

COpenGLOcclusionQuery::COpenGLOcclusionQuery(COpenGLDriver* driver_in, GLenum type_in) : object(0), counter(2147483647), driver(driver_in), type(type_in), condModeGL(GL_QUERY_NO_WAIT)
{
    driver->extGlGenQueries(1,&object);
    driver->extGlBeginQuery(type,object); // init the object
    driver->extGlEndQuery(type);
}

COpenGLOcclusionQuery::~COpenGLOcclusionQuery()
{
    driver->extGlDeleteQueries(1,&object);
}

u32 COpenGLOcclusionQuery::getOcclusionQueryResult() const
{
    if (type==GL_ANY_SAMPLES_PASSED)
        return counter>=1 ? (~0u):0;
    else
        return counter;
}

GLenum COpenGLOcclusionQuery::getType() const
{
    return type;
}

GLuint COpenGLOcclusionQuery::getGLHandle() const
{
    return object;
}

void COpenGLOcclusionQuery::setCondWaitMode(const E_CONDITIONAL_RENDERING_WAIT_MODE& mode)
{
    IOcclusionQuery::setCondWaitMode(mode);

    switch (mode)
    {
        case ECRWM_WAIT:
            condModeGL = GL_QUERY_WAIT;
            break;
        default:
            condModeGL = GL_QUERY_NO_WAIT;
            break;
    }
}


}
}
