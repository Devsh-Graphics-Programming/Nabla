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

class Uncopyable
{
    public:
        _IRR_NO_COPY_FINAL(TYPE);
};

class Interface : public Uncopyable
{
        _IRR_NO_PUBLIC_DELETE_DEFAULT(TYPE);
};

class Unmovable
{
    public:
        _IRR_NO_MOVE_FINAL(TYPE);
};

class InterfaceUnmovable : public Interface, public Unmovable
{
        _IRR_INTERFACE_CHILD_DEFAULT(TYPE);
};

class TotalInterface : public Interface
{
        _IRR_INTERFACE_CHILD_DEFAULT(TYPE);
    public:
        _IRR_NO_DEFAULT_FINAL(TYPE);
};

class TotalInterfaceUnmovable : public TotalInterface, public Unmovable
{
        _IRR_INTERFACE_CHILD_DEFAULT(TYPE);
};


/** TODO: Classes for objects requiring memory alignment
1) Declare alignment on the parent class through a template (hoepfully alignment is inherited in C++11)
2) Define a macro to delete the new and delete operators in a class OR override new/delete globally
3) Also maybe override malloc, calloc, realloc and free
**/

#endif
