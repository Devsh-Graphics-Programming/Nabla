// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_BASE_CLASSES_H_INCLUDED__
#define __IRR_BASE_CLASSES_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "irrMacros.h"

#define _IRR_INTERFACE_CHILD(TYPE) \
            _IRR_NO_PUBLIC_DELETE(TYPE)

#define _IRR_INTERFACE_CHILD_DEFAULT(TYPE) \
            _IRR_NO_PUBLIC_DELETE_DEFAULT(TYPE)

namespace irr
{

class Uncopyable
{
    public:
        Uncopyable() = default;
        _IRR_NO_COPY_FINAL(Uncopyable);
};

class Interface : public Uncopyable
{
        _IRR_NO_PUBLIC_DELETE_DEFAULT(Interface);
    protected:
        Interface() = default;
};

class Unmovable
{
    public:
        Unmovable() = default;
        _IRR_NO_MOVE_FINAL(Unmovable);
};

class InterfaceUnmovable : public Interface, public Unmovable
{
        _IRR_INTERFACE_CHILD_DEFAULT(InterfaceUnmovable);
    public:
        InterfaceUnmovable() = default;
};

class TotalInterface : public Interface
{
        _IRR_INTERFACE_CHILD_DEFAULT(TotalInterface);
    public:
        _IRR_NO_DEFAULT_FINAL(TotalInterface);
};

class TotalInterfaceUnmovable : public TotalInterface, public Unmovable
{
        _IRR_INTERFACE_CHILD_DEFAULT(TotalInterfaceUnmovable);
};

}

/** TODO: Classes for objects requiring memory alignment
1) Declare alignment on the parent class through a template (hoepfully alignment is inherited in C++11)
2) Define a macro to delete the new and delete operators in a class OR override new/delete globally
3) Also maybe override malloc, calloc, realloc and free
**/

#endif
