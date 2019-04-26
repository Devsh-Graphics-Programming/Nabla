#include "COpenGLTransformFeedback.h"
#include "os.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
using namespace irr;
using namespace video;

COpenGLTransformFeedback::COpenGLTransformFeedback() : xformFeedbackHandle(0), cachedPrimitiveType(GL_INVALID_ENUM), cachedXFormFeedbackShader(-1), lastValidated(0), started(false)
{
    for (size_t i=0; i<_IRR_XFORM_FEEDBACK_MAX_BUFFERS_; i++)
    {
        xformFeedbackBuffers[i] = NULL;
        xformFeedbackBuffersOffset[i] = 0;
        xformFeedbackBuffersSize[i] = 0;
    }

    COpenGLExtensionHandler::extGlCreateTransformFeedbacks(1,&xformFeedbackHandle);
}

COpenGLTransformFeedback::~COpenGLTransformFeedback()
{
    COpenGLExtensionHandler::extGlDeleteTransformFeedbacks(1,&xformFeedbackHandle);

    for (size_t i=0; i<_IRR_XFORM_FEEDBACK_MAX_BUFFERS_; i++)
    {
        if (xformFeedbackBuffers[i])
            xformFeedbackBuffers[i]->drop();
    }
}

bool COpenGLTransformFeedback::rebindRevalidate()
{
    uint64_t highestRevalidateStamp = lastValidated;

    for (size_t i=0; i<_IRR_XFORM_FEEDBACK_MAX_BUFFERS_; i++)
    {
        if (!xformFeedbackBuffers[i])
            continue;

        COpenGLBuffer* buffer = xformFeedbackBuffers[i];

        uint64_t revalidateStamp = buffer->getLastTimeReallocated();
        if (revalidateStamp>lastValidated)
        {
            if (revalidateStamp>highestRevalidateStamp)
                highestRevalidateStamp = revalidateStamp;

            if (xformFeedbackBuffersOffset[i]||(xformFeedbackBuffersSize[i]+xformFeedbackBuffersOffset[i])<buffer->getSize())
                COpenGLExtensionHandler::extGlTransformFeedbackBufferRange(xformFeedbackHandle,i,buffer->getOpenGLName(),xformFeedbackBuffersOffset[i],xformFeedbackBuffersSize[i]);
            else
                COpenGLExtensionHandler::extGlTransformFeedbackBufferBase(xformFeedbackHandle,i,buffer->getOpenGLName());
        }
    }

    lastValidated = highestRevalidateStamp;

	return true;
}

bool COpenGLTransformFeedback::bindOutputBuffer(const size_t& index, IGPUBuffer* buffer, const size_t& offset, const size_t& length)
{
    if (index>=_IRR_XFORM_FEEDBACK_MAX_BUFFERS_)
        return false;

    if (offset&0x3u)
    {
#ifdef _IRR_DEBUG
//        FW_WriteToLog(kLogError,"XFormFeedback Buffer Offset is Un-aligned to 4 bytes! Binding failure!\n");
#endif // _IRR_DEBUG
        return false;
    }

    size_t tmpSize;
    if (buffer)
        tmpSize = length>0 ? length:(buffer->getSize()-offset);


    //no chance needed if all parameters are equal, or buffers are equal and NULL
    if (xformFeedbackBuffers[index]==buffer)
    {
        if (buffer==NULL||(xformFeedbackBuffersOffset[index]==offset&&xformFeedbackBuffersSize[index]==tmpSize))
            return true;
    }
    else if (buffer) //if binding a new buffer
    {
        buffer->grab();
        if (xformFeedbackBuffers[index])
            xformFeedbackBuffers[index]->drop();
    }
    else //just unbinding old buffer
    {
        if (xformFeedbackBuffers[index])
            xformFeedbackBuffers[index]->drop();
        xformFeedbackBuffers[index] = NULL;
        xformFeedbackBuffersOffset[index] = 0;
        xformFeedbackBuffersSize[index] = 0;
        COpenGLExtensionHandler::extGlTransformFeedbackBufferBase(xformFeedbackHandle,index,0);
        return true;
    }

    xformFeedbackBuffers[index] = static_cast<COpenGLBuffer*>(buffer);
    xformFeedbackBuffersOffset[index] = offset;
    xformFeedbackBuffersSize[index] = tmpSize;


    if (xformFeedbackBuffersOffset[index]||(xformFeedbackBuffersSize[index]+xformFeedbackBuffersOffset[index])<buffer->getSize())
        COpenGLExtensionHandler::extGlTransformFeedbackBufferRange(xformFeedbackHandle,index,xformFeedbackBuffers[index]->getOpenGLName(),xformFeedbackBuffersOffset[index],xformFeedbackBuffersSize[index]);
    else
        COpenGLExtensionHandler::extGlTransformFeedbackBufferBase(xformFeedbackHandle,index,xformFeedbackBuffers[index]->getOpenGLName());

    return true;
}


void COpenGLTransformFeedback::beginResumeFeedback()
{
#ifdef _IRR_DEBUG
    if (active)
    {
        os::Printer::log("Trying to resume an already active Transform Feedback!\n",ELL_ERROR);
        return;
    }
#endif
    if (!rebindRevalidate())
    {
        os::Printer::log("COpenGLTransformFeedback::RebindRevalidate FAILED!\n",ELL_ERROR);
        return;
    }

    if (started)
        COpenGLExtensionHandler::extGlResumeTransformFeedback();
    else
    {
        COpenGLExtensionHandler::extGlBeginTransformFeedback(cachedPrimitiveType);
        started = true;
    }
    active = true;
}

void COpenGLTransformFeedback::pauseFeedback()
{
#ifdef _IRR_DEBUG
    if (!active)
        os::Printer::log("Trying to pause an inactive feedback!\n",ELL_ERROR);
#endif
    COpenGLExtensionHandler::extGlPauseTransformFeedback();
    active = false;
}

void COpenGLTransformFeedback::endFeedback()
{
#ifdef _IRR_DEBUG
    if (!started)
        os::Printer::log("Trying to End an un-started Feedback Transform!\n",ELL_ERROR);
#endif
    COpenGLExtensionHandler::extGlEndTransformFeedback();
    started = false;
    active = false;
}
#endif
