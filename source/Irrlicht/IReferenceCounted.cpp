#include "irr/core/IReferenceCounted.h"

using namespace irr;
using namespace core;

IReferenceCounted::~IReferenceCounted()
{
    _IRR_DEBUG_BREAK_IF(ReferenceCounter!=0);
}
