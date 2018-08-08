// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_BASE_CLASSES_H_INCLUDED__
#define __IRR_BASE_CLASSES_H_INCLUDED__

#include "irrMemory.h"

#define _IRR_INTERFACE_CHILD(TYPE) \
            _IRR_NO_PUBLIC_DELETE(TYPE)

#define _IRR_INTERFACE_CHILD_DEFAULT(TYPE) \
            _IRR_NO_PUBLIC_DELETE_DEFAULT(TYPE)

namespace irr
{

class FORCE_EMPTY_BASE_OPT Uncopyable
{
    public:
        Uncopyable() = default;
        _IRR_NO_COPY_FINAL(Uncopyable);
};

class FORCE_EMPTY_BASE_OPT Interface : public Uncopyable
{
        _IRR_NO_PUBLIC_DELETE_DEFAULT(Interface);
    protected:
        Interface() = default;
};

class FORCE_EMPTY_BASE_OPT Unmovable
{
    public:
        Unmovable() = default;
        _IRR_NO_MOVE_FINAL(Unmovable);
};

class FORCE_EMPTY_BASE_OPT InterfaceUnmovable : public Interface, public Unmovable
{
        _IRR_INTERFACE_CHILD_DEFAULT(InterfaceUnmovable);
    public:
        InterfaceUnmovable() = default;
};

class FORCE_EMPTY_BASE_OPT TotalInterface : public Interface
{
        _IRR_INTERFACE_CHILD_DEFAULT(TotalInterface);
    public:
        _IRR_NO_DEFAULT_FINAL(TotalInterface);
};

class FORCE_EMPTY_BASE_OPT TotalInterfaceUnmovable : public TotalInterface, public Unmovable
{
        _IRR_INTERFACE_CHILD_DEFAULT(TotalInterfaceUnmovable);
};

}

#endif
