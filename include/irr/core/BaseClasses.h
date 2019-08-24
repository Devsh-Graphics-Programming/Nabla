// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_BASE_CLASSES_H_INCLUDED__
#define __IRR_BASE_CLASSES_H_INCLUDED__

#include "irr/core/memory/memory.h"

#define _IRR_INTERFACE_CHILD(TYPE) \
            _IRR_NO_PUBLIC_DELETE(TYPE)

#define _IRR_INTERFACE_CHILD_DEFAULT(TYPE) \
            _IRR_NO_PUBLIC_DELETE_DEFAULT(TYPE)

namespace irr
{
namespace core
{

class IRR_FORCE_EBO IRR_NO_VTABLE Uncopyable
{
    public:
        Uncopyable() = default;
        _IRR_NO_COPY_FINAL(Uncopyable);
};

class IRR_FORCE_EBO IRR_NO_VTABLE Interface : public Uncopyable
{
        _IRR_NO_PUBLIC_DELETE_DEFAULT(Interface);
    protected:
        Interface() = default;
};

class IRR_FORCE_EBO IRR_NO_VTABLE Unmovable
{
    public:
        Unmovable() = default;
        _IRR_NO_MOVE_FINAL(Unmovable);
};

class IRR_FORCE_EBO IRR_NO_VTABLE InterfaceUnmovable : public Interface, public Unmovable
{
        _IRR_INTERFACE_CHILD_DEFAULT(InterfaceUnmovable);
    public:
        InterfaceUnmovable() = default;
};

class IRR_FORCE_EBO IRR_NO_VTABLE TotalInterface : public Interface
{
        _IRR_INTERFACE_CHILD_DEFAULT(TotalInterface);
    public:
        _IRR_NO_DEFAULT_FINAL(TotalInterface);
};

class IRR_FORCE_EBO IRR_NO_VTABLE TotalInterfaceUnmovable : public TotalInterface, public Unmovable
{
        _IRR_INTERFACE_CHILD_DEFAULT(TotalInterfaceUnmovable);
};

}
}

#endif
