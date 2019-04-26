#include "COpenGLQuery.h"
#include "COpenGLBuffer.h"
#include "COpenGLExtensionHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
namespace irr
{
namespace video
{

COpenGLQuery::COpenGLQuery(const GLenum& type_in) :  object(0), type(type_in), active(false), queryNeedsUpdate(false), queryIsReady(true), cachedCounter32(0), cachedCounter64(0)
{
    COpenGLExtensionHandler::extGlCreateQueries(type,1,&object);

    switch (type)
    {
        case GL_PRIMITIVES_GENERATED:
            cachedIrrType = EQOT_PRIMITIVES_GENERATED;
            break;
        case GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN:
            cachedIrrType = EQOT_XFORM_FEEDBACK_PRIMITIVES_WRITTEN;
            break;
        case GL_TIME_ELAPSED:
            cachedIrrType = EQOT_TIME_ELAPSED;
            break;
        default:
            cachedIrrType = EQOT_COUNT;
            break;
    }
}

COpenGLQuery::~COpenGLQuery()
{
    COpenGLExtensionHandler::extGlDeleteQueries(1,&object);
}

void COpenGLQuery::updateQueryResult()
{
    if (queryNeedsUpdate)
    {
        switch (cachedIrrType)
        {
            case EQOT_PRIMITIVES_GENERATED:
            case EQOT_XFORM_FEEDBACK_PRIMITIVES_WRITTEN:
                COpenGLExtensionHandler::extGlGetQueryObjectuiv(object,GL_QUERY_RESULT,&cachedCounter32);
                cachedCounter64 = cachedCounter32;
                break;
            default:
                COpenGLExtensionHandler::extGlGetQueryObjectui64v(object,GL_QUERY_RESULT,&cachedCounter64);
                cachedCounter32 = cachedCounter64;
                break;
        }
    }
    queryNeedsUpdate = false;
    queryIsReady = true;
}

void COpenGLQuery::getQueryResult(uint32_t* queryResult)
{
#ifdef _IRR_DEBUG
    if (active)
    {
        os::Printer::log("CANNOT FETCH QUERY RESULT WHILE THE QUERY IS RUNNING!\n",ELL_ERROR);
        return;
    }
#endif // _IRR_DEBUG

    updateQueryResult();
    *queryResult = cachedCounter32;
}

void COpenGLQuery::getQueryResult(uint64_t* queryResult)
{
#ifdef _IRR_DEBUG
    if (active)
    {
        os::Printer::log("CANNOT FETCH QUERY RESULT WHILE THE QUERY IS RUNNING!\n",ELL_ERROR);
        return;
    }
#endif // _IRR_DEBUG

    updateQueryResult();
    *queryResult = cachedCounter64;
}

bool COpenGLQuery::getQueryResult32(IGPUBuffer* buffer, const size_t& offset, const bool& conditionalWrite)
{
#ifdef _IRR_DEBUG
    if (active)
    {
        os::Printer::log("CANNOT FETCH QUERY RESULT WHILE THE QUERY IS RUNNING!\n",ELL_ERROR);
        return false;
    }
#endif // _IRR_DEBUG

    COpenGLBuffer* asGLBuf = static_cast<COpenGLBuffer*>(buffer);
    if (!asGLBuf)
    {
#ifdef _IRR_DEBUG
        os::Printer::log("CANNOT FETCH QUERY RESULT, buffer is NULL or not OpenGLBuffer!\n",ELL_ERROR);
#endif // _IRR_DEBUG
        return false;
    }

    COpenGLExtensionHandler::extGlGetQueryBufferObjectuiv(object,asGLBuf->getOpenGLName(),conditionalWrite ? GL_QUERY_RESULT_NO_WAIT:GL_QUERY_RESULT,offset);
    return COpenGLExtensionHandler::Version>=440&&COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_ARB_query_buffer_object];
}

bool COpenGLQuery::getQueryResult64(IGPUBuffer* buffer, const size_t& offset, const bool& conditionalWrite)
{
#ifdef _IRR_DEBUG
    if (active)
    {
        os::Printer::log("CANNOT FETCH QUERY RESULT WHILE THE QUERY IS RUNNING!\n",ELL_ERROR);
        return false;
    }
#endif // _IRR_DEBUG

    COpenGLBuffer* asGLBuf = static_cast<COpenGLBuffer*>(buffer);
    if (!asGLBuf)
    {
#ifdef _IRR_DEBUG
        os::Printer::log("CANNOT FETCH QUERY RESULT, buffer is NULL or not OpenGLBuffer!\n",ELL_ERROR);
#endif // _IRR_DEBUG
        return false;
    }

    COpenGLExtensionHandler::extGlGetQueryBufferObjectui64v(object,asGLBuf->getOpenGLName(),conditionalWrite ? GL_QUERY_RESULT_NO_WAIT:GL_QUERY_RESULT,offset);
    return COpenGLExtensionHandler::Version>=440&&COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_ARB_query_buffer_object];
}


bool COpenGLQuery::isQueryReady()
{
    if (active)
        return false;

    if (!queryNeedsUpdate || queryIsReady)
        return true;

    GLuint available = GL_FALSE;
    COpenGLExtensionHandler::extGlGetQueryObjectuiv(object,GL_QUERY_RESULT_AVAILABLE,&available);
    queryIsReady = available!=GL_FALSE;

    return queryIsReady;
}

void COpenGLQuery::isQueryReady32(IGPUBuffer* buffer, const size_t& offset)
{
#ifdef _IRR_DEBUG
    if (active)
    {
        os::Printer::log("CANNOT FETCH QUERY RESULT WHILE THE QUERY IS RUNNING!\n",ELL_ERROR);
        return;
    }
#endif // _IRR_DEBUG

    COpenGLBuffer* asGLBuf = static_cast<COpenGLBuffer*>(buffer);
    if (!asGLBuf)
    {
#ifdef _IRR_DEBUG
        os::Printer::log("CANNOT FETCH QUERY RESULT, buffer is NULL or not OpenGLBuffer!\n",ELL_ERROR);
#endif // _IRR_DEBUG
        return;
    }

    COpenGLExtensionHandler::extGlGetQueryBufferObjectuiv(object,asGLBuf->getOpenGLName(),GL_QUERY_RESULT_AVAILABLE,offset);
}

void COpenGLQuery::isQueryReady64(IGPUBuffer* buffer, const size_t& offset)
{
    //if (COpenGLExtensionHandler::Version<440 && !COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_query_buffer_object])
#ifdef _IRR_DEBUG
    if (active)
    {
        os::Printer::log("CANNOT FETCH QUERY RESULT WHILE THE QUERY IS RUNNING!\n",ELL_ERROR);
        return;
    }
#endif // _IRR_DEBUG

    COpenGLBuffer* asGLBuf = static_cast<COpenGLBuffer*>(buffer);
    if (!asGLBuf)
    {
#ifdef _IRR_DEBUG
        os::Printer::log("CANNOT FETCH QUERY RESULT, buffer is NULL or not OpenGLBuffer!\n",ELL_ERROR);
#endif // _IRR_DEBUG
        return;
    }

    COpenGLExtensionHandler::extGlGetQueryBufferObjectui64v(object,asGLBuf->getOpenGLName(),GL_QUERY_RESULT_AVAILABLE,offset);
}


}
}
#endif // _IRR_COMPILE_WITH_OPENGL_
