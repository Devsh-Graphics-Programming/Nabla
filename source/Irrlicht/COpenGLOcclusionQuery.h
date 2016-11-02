#ifndef COPENGLOCCLUSIONQUERY_H
#define COPENGLOCCLUSIONQUERY_H

#include "IOcclusionQuery.h"
#ifdef _IRR_COMPILE_WITH_OPENGL_
#include "COpenGLQuery.h"

namespace irr
{
namespace video
{

class COpenGLOcclusionQuery : public IOcclusionQuery, public COpenGLQuery
{
    public:
        COpenGLOcclusionQuery(const E_OCCLUSION_QUERY_TYPE& heuristic);
        //virtual ~COpenGLOcclusionQuery() {}

		virtual void setCondWaitMode(const E_CONDITIONAL_RENDERING_WAIT_MODE& mode);
		inline GLenum getCondWaitModeGL() const {return condModeGL;}



        /// ALL will STALL CPU IF QUERY NOT READY
		virtual void getQueryResult(uint32_t* queryResult) {COpenGLQuery::getQueryResult(queryResult);}
		virtual void getQueryResult(uint64_t* queryResult) {COpenGLQuery::getQueryResult(queryResult);}
		//except for this one, maybe... depends on driver, can still stall GPU
		/// AVAILABLE ONLY WITH ARB_query_buffer_object !!
		virtual bool getQueryResult32(IGPUBuffer* buffer, const size_t& offset=0, const bool& conditionalWrite=true) {return COpenGLQuery::getQueryResult32(buffer,offset,conditionalWrite);}
		virtual bool getQueryResult64(IGPUBuffer* buffer, const size_t& offset=0, const bool& conditionalWrite=true) {return COpenGLQuery::getQueryResult64(buffer,offset,conditionalWrite);}

		virtual bool isQueryReady() {return COpenGLQuery::isQueryReady();}
		//included only to expose full GL capabilities
		/// AVAILABLE ONLY WITH ARB_query_buffer_object !!
		virtual void isQueryReady32(IGPUBuffer* buffer, const size_t& offset=0) {return COpenGLQuery::isQueryReady32(buffer,offset);}
		virtual void isQueryReady64(IGPUBuffer* buffer, const size_t& offset=0) {return COpenGLQuery::isQueryReady64(buffer,offset);}

		virtual const E_QUERY_OBJECT_TYPE getQueryObjectType() const {return EQOT_OCCLUSION;}
    protected:
    private:
        GLenum condModeGL;
};


}
}
#endif // _IRR_COMPILE_WITH_OPENGL_

#endif // COPENGLOCCLUSIONQUERY_H
