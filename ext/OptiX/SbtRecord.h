#ifndef __IRR_EXT_SBT_RECORD_H_INCLUDED__
#define __IRR_EXT_SBT_RECORD_H_INCLUDED__

#include "optix.h"

namespace irr
{
namespace ext
{
namespace OptiX
{

template <typename T>
struct SbtRecord
{
	alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

}
}
}

#endif