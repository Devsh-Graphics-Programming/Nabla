#include "COpenGLOcclusionQuery.h"

namespace irr
{
namespace video
{

COpenGLOcclusionQuery::COpenGLOcclusionQuery(const E_OCCLUSION_QUERY_TYPE& heuristic)
                        : IOcclusionQuery(heuristic), COpenGLQuery(heuristic==EOQT_SAMPLES_PASSED ? GL_SAMPLES_PASSED:(heuristic==EOQT_ANY_SAMPLES_PASSED ? GL_ANY_SAMPLES_PASSED:GL_ANY_SAMPLES_PASSED_CONSERVATIVE)),
                        condModeGL(GL_QUERY_NO_WAIT)
{
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
