// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_QUERY_OBJECT_H_INCLUDED__
#define __NBL_I_QUERY_OBJECT_H_INCLUDED__

#include "nbl/core/decl/Types.h"
#include "nbl/core/IReferenceCounted.h"

namespace nbl
{
namespace video
{

class IGPUBuffer;

enum E_QUERY_OBJECT_TYPE
{
    EQOT_PRIMITIVES_GENERATED=0u,
    EQOT_XFORM_FEEDBACK_PRIMITIVES_WRITTEN,
    EQOT_TIME_ELAPSED,
    //EQOT_TIMESTAMP,
    EQOT_COUNT
};

class IQueryObject : public core::IReferenceCounted
{
	    _NBL_INTERFACE_CHILD(IQueryObject) {}
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

#endif

