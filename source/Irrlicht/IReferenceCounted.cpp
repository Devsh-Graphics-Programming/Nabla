#include "IReferenceCounted.h"

using namespace irr;

IReferenceCounted::~IReferenceCounted()
{
    _IRR_DEBUG_BREAK_IF(ReferenceCounter!=0);
}
