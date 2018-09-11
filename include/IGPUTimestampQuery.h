#ifndef __I_GPU_TIMESTAMP_QUERY_H_INCLUDED__
#define __I_GPU_TIMESTAMP_QUERY_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "stdint.h"

namespace irr
{
namespace video
{


class IGPUTimestampQuery : public core::IReferenceCounted
{
	    _IRR_INTERFACE_CHILD(IGPUTimestampQuery) {}
    public:
		virtual bool isQueryReady() = 0;

        virtual uint64_t getTimestampWhenCompleted() = 0;
};

}
}

#endif // __I_GPU_TIMESTAMP_QUERY_H_INCLUDED__


