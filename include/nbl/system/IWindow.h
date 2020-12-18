#ifndef __NBL_I_WINDOW_H_INCLUDED__
#define __NBL_I_WINDOW_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace system
{

class IWindow : public core::IReferenceCounted
{
public:
    virtual uint32_t getWidth() const = 0;
    virtual uint32_t getHeight() const = 0;
};

}
}


#endif