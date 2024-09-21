/* nabla.h -- interface of the 'Nabla Engine'

  Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.

  This software is provided 'as-is', under the Apache 2.0 license,
  without any express or implied warranty.  In no event will the authors
  be held liable for any damages arising from the use of this software.

  See LICENSE.md for full licensing information.

  Please note that the IrrlichtBAW Engine is based in part on the work of others,
  this means that if you use the IrrlichtBAW Engine in your product,
  you must acknowledge somewhere in your documentation that you've used their code.
  See README.md for all mentions of 3rd party software used.
*/

#ifndef __NABLA_H_INCLUDED__
#define __NABLA_H_INCLUDED__

// meta info
#include "git_info.h"

namespace nbl {
	NBL_API2 const gtml::GitInfo& getGitInfo(gtml::E_GIT_REPO_META repo);
}

// core lib
#include "nbl/core/declarations.h"

// system lib (fibers, mutexes, file I/O operations) [DEPENDS: core]
#include "nbl/system/declarations.h"
// TODO: should we move "core/parallel" to "system/parallel"

// asset lib (importing and exporting meshes, textures and shaders) [DEPENDS: system]
#include "nbl/asset/asset.h"

// ui lib (window set up, software blit, joysticks, multi-touch, keyboard, etc.) [DEPENDS: system]
#include "nbl/ui/declarations.h"

// video lib (access to Graphics API, remote rendering, etc) [DEPENDS: asset, (optional) ui]
#include "nbl/video/declarations.h"

// scene lib (basic rendering, culling, scene graph etc.) [DEPENDS: video, ui]
#include "nbl/scene/scene.h"

// core lib
#include "nbl/core/definitions.h"

// system lib (fibers, mutexes, file I/O operations) [DEPENDS: core]
#include "nbl/system/definitions.h"

// ui lib (window set up, software blit, joysticks, multi-touch, keyboard, etc.) [DEPENDS: system]
#include "nbl/ui/definitions.h"

// video lib (access to Graphics API, remote rendering, etc) [DEPENDS: asset, (optional) ui]
#include "nbl/video/definitions.h"

#include "aabbox3d.h"
#include "vector2d.h"
#include "vector3d.h"
#include "vectorSIMD.h"
#include "line3d.h"
#include "matrix4SIMD.h"
#include "position2d.h"
#include "quaternion.h"
#include "rect.h"
#include "dimension2d.h"

// TEST

#include "splines.h"

#include "SColor.h"

#endif // __NABLA_H_INCLUDED__