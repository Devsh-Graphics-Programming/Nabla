#ifndef __I_IOCCLUSION_QUERY_H_INCLUDED__
#define __I_IOCCLUSION_QUERY_H_INCLUDED__

#include <IQueryObject.h>


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

enum E_OCCLUSION_QUERY_TYPE
{
    EOQT_SAMPLES_PASSED=0,
    EOQT_ANY_SAMPLES_PASSED,
    EOQT_ANY_SAMPLES_PASSED_CONSERVATIVE,
    EOQT_COUNT
};

class IOcclusionQuery : public virtual IQueryObject
{
    public:
		inline E_CONDITIONAL_RENDERING_WAIT_MODE getCondWaitMode() const {return waitMode;}
		virtual void setCondWaitMode(const E_CONDITIONAL_RENDERING_WAIT_MODE& mode) {waitMode = mode;}

		virtual E_QUERY_OBJECT_TYPE getQueryObjectType() const {return EQOT_OCCLUSION;}

		const E_OCCLUSION_QUERY_TYPE& getOcclusionQueryType() const {return oqt;}
    protected:
        IOcclusionQuery(const E_OCCLUSION_QUERY_TYPE& occlusionQueryTypeHeuristic) : oqt(occlusionQueryTypeHeuristic) {}
    private:
        E_OCCLUSION_QUERY_TYPE oqt;
        E_CONDITIONAL_RENDERING_WAIT_MODE waitMode;
};

}
}

#endif // __I_IOCCLUSION_QUERY_H_INCLUDED__
