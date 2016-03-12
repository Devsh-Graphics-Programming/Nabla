#ifndef __I_IOCCLUSION_QUERY_H_INCLUDED__
#define __I_IOCCLUSION_QUERY_H_INCLUDED__

#include <IReferenceCounted.h>

/// SHOULD REALLY BE CALLED IGPUCounterQuery
/**
Since we can have the following queries:
GL_TIMESTAMP
GL_TIME_ELAPSED
GL_TRANSFORM_FEEDBACK
etc.
**/

namespace irr
{
namespace video
{

enum E_CONDITIONAL_RENDERING_WAIT_MODE
{
    ECRWM_WAIT=0,
    ECRWM_NO_WAIT,
    ECRWM_COUNT
};

class IOcclusionQuery : public IReferenceCounted
{
    public:
        IOcclusionQuery() {}
        virtual ~IOcclusionQuery() {}
		virtual u32 getOcclusionQueryResult() const = 0;
		virtual E_CONDITIONAL_RENDERING_WAIT_MODE getCondWaitMode() const {return waitMode;}
		virtual void setCondWaitMode(const E_CONDITIONAL_RENDERING_WAIT_MODE& mode) {waitMode = mode;}
    protected:
    private:
        E_CONDITIONAL_RENDERING_WAIT_MODE waitMode;
};

}
}

#endif // __I_IOCCLUSION_QUERY_H_INCLUDED__
