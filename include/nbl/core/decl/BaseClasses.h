// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef __NBL_CORE_BASE_CLASSES_H_INCLUDED__
#define __NBL_CORE_BASE_CLASSES_H_INCLUDED__

#include "nbl/core/memory/memory.h"

#define _NBL_INTERFACE_CHILD(TYPE) \
            _NBL_NO_PUBLIC_DELETE(TYPE)

#define _NBL_INTERFACE_CHILD_DEFAULT(TYPE) \
            _NBL_NO_PUBLIC_DELETE_DEFAULT(TYPE)

namespace nbl::core
{

class NBL_FORCE_EBO NBL_NO_VTABLE Uncopyable
{
    public:
        Uncopyable() = default;
        _NBL_NO_COPY_FINAL(Uncopyable);
};

class NBL_FORCE_EBO NBL_NO_VTABLE Interface : public Uncopyable
{
        _NBL_NO_PUBLIC_DELETE_DEFAULT(Interface);
    protected:
        Interface() = default;
};

class NBL_FORCE_EBO NBL_NO_VTABLE Unmovable
{
    public:
        Unmovable() = default;
        _NBL_NO_MOVE_FINAL(Unmovable);
};

class NBL_FORCE_EBO NBL_NO_VTABLE InterfaceUnmovable : public Interface, public Unmovable
{
        _NBL_INTERFACE_CHILD_DEFAULT(InterfaceUnmovable);
    public:
        InterfaceUnmovable() = default;
};

class NBL_FORCE_EBO NBL_NO_VTABLE TotalInterface : public Interface
{
        _NBL_INTERFACE_CHILD_DEFAULT(TotalInterface);
    public:
        _NBL_NO_DEFAULT_FINAL(TotalInterface);
};

class NBL_FORCE_EBO NBL_NO_VTABLE TotalInterfaceUnmovable : public TotalInterface, public Unmovable
{
        _NBL_INTERFACE_CHILD_DEFAULT(TotalInterfaceUnmovable);
};


}

#endif
