#ifndef __I_QUERY_OBJECT_H_INCLUDED__
#define __I_QUERY_OBJECT_H_INCLUDED__

#include <IReferenceCounted.h>
#include "stdint.h"

namespace irr
{
namespace video
{

class IGPUBuffer;

enum E_QUERY_OBJECT_TYPE
{
    EQOT_OCCLUSION=0,
    EQOT_PRIMITIVES_GENERATED,
    EQOT_XFORM_FEEDBACK_PRIMITIVES_WRITTEN,
    EQOT_TIME_ELAPSED,
    //EQOT_TIMESTAMP,
    EQOT_COUNT
};

class IQueryObject : public IReferenceCounted
{
    public:
        /// ALL will STALL CPU IF QUERY NOT READY
		virtual void getQueryResult(uint32_t* queryResult) = 0;
		virtual void getQueryResult(uint64_t* queryResult) = 0;
		//except for this one, maybe... depends on driver, can still stall GPU
		/// AVAILABLE ONLY WITH ARB_query_buffer_object !!
		virtual bool getQueryResult32(IGPUBuffer* buffer, const size_t& offset=0, const bool& conditionalWrite=true) = 0;
		virtual bool getQueryResult64(IGPUBuffer* buffer, const size_t& offset=0, const bool& conditionalWrite=true) = 0;

		virtual bool isQueryReady() = 0;
		//included only to expose full GL capabilities
		/// AVAILABLE ONLY WITH ARB_query_buffer_object !!
		virtual void isQueryReady32(IGPUBuffer* buffer, const size_t& offset=0) = 0;
		virtual void isQueryReady64(IGPUBuffer* buffer, const size_t& offset=0) = 0;

		virtual E_QUERY_OBJECT_TYPE getQueryObjectType() const =0 ;
};

}
}

#endif // __I_QUERY_OBJECT_H_INCLUDED__

