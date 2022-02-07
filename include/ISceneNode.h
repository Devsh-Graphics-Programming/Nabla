// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_SCENE_NODE_H_INCLUDED__
#define __NBL_I_SCENE_NODE_H_INCLUDED__

#include "nbl/video/video.h"

#include "ISceneNodeAnimator.h"
#include "aabbox3d.h"
#include "matrix4x3.h"
#include "IDummyTransformationSceneNode.h"
#include "IDriverFence.h"

namespace nbl
{
namespace scene
{
class ISceneManager;

//! Scene node interface.
/** A scene node is a node in the hierarchical scene tree. Every scene
	node may have children, which are also scene nodes. Children move
	relative to their parent's position. If the parent of a node is not
	visible, its children won't be visible either. In this way, it is for
	example easily possible to attach a light to a moving car, or to place
	a walking character on a moving platform on a moving ship.
	*/
class ISceneNode : public IDummyTransformationSceneNode
{
public:
    //! Constructor
    ISceneNode(IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id = -1,
        const core::vector3df& position = core::vector3df(0, 0, 0),
        const core::vector3df& rotation = core::vector3df(0, 0, 0),
        const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f))
        : IDummyTransformationSceneNode(parent, position, rotation, scale),
          AutomaticCullingState(true), IsVisible(true)
    {
    }

    virtual bool isISceneNode() const { return true; }

    //! OnAnimate() is called just before rendering the whole scene.
    /** Nodes may calculate or store animations here, and may do other useful things,
		depending on what they are. Also, OnAnimate() should be called for all
		child scene nodes here. This method will be called once per frame, independent
		of whether the scene node is visible or not.
		\param timeMs Current time in milliseconds. */
    virtual void OnAnimate(uint32_t timeMs)
    {
        OnAnimate_static(this, timeMs);
    }

    //! Renders the node.
    virtual void render() = 0;

    //! Returns whether the node should be visible (only matters if all of its parents are visible).
    /** This is only an option set by the user, but has nothing to
		do with geometry culling
		\return The requested visibility of the node, true means
		visible (if all parents are also visible). */
    virtual bool isVisible() const
    {
        return IsVisible;
    }

    //! Check whether the node is truly visible, taking into accounts its parents' visibility
    /** \return true if the node and all its parents are visible,
		false if this or any parent node is invisible. */
    virtual bool isTrulyVisible() const
    {
        return isTrulyVisible_static(this);
    }

    //! Sets if the node should be visible or not.
    /** All children of this node won't be visible either, when set
		to false. Invisible nodes are not valid candidates for selection by
		collision manager bounding box methods.
		\param isVisible If the node shall be visible. */
    virtual void setVisible(bool isVisible_in)
    {
        IsVisible = isVisible_in;
    }

protected:
    //! Destructor
    virtual ~ISceneNode()
    {
    }

    //! Automatic culling state
    bool AutomaticCullingState;

    //! Is the node visible?
    bool IsVisible;

    static void OnAnimate_static(IDummyTransformationSceneNode* node, uint32_t timeMs)  // could be pushed up to IDummyTransformationSceneNode
    {
        ISceneNode* tmp = static_cast<ISceneNode*>(node);
        if(!node->isISceneNode() || tmp->IsVisible)
        {
            // animate this node with all animators

            //! The bloody animator can remove itself during animateNode!!!!
            const ISceneNodeAnimatorArray& animators = node->getAnimators();
            size_t prevSize = animators.size();
            for(size_t i = 0; i < prevSize;)
            {
                ISceneNodeAnimator* anim = animators[i];
                anim->animateNode(node, timeMs);
                if(animators[i] > anim)
                    prevSize = animators.size();
                else
                    i++;
            }

            // update absolute position
            node->updateAbsolutePosition();

            // perform the post render process on all children
            const IDummyTransformationSceneNodeArray& children = node->getChildren();
            prevSize = children.size();
            for(size_t i = 0; i < prevSize;)
            {
                IDummyTransformationSceneNode* tmpChild = children[i];
                if(tmpChild->isISceneNode())
                    static_cast<ISceneNode*>(tmpChild)->OnAnimate(timeMs);
                else
                    OnAnimate_static(tmpChild, timeMs);

                if(children[i] > tmpChild)
                    prevSize = children.size();
                else
                    i++;
            }
        }
    }

private:
    static bool isTrulyVisible_static(const IDummyTransformationSceneNode* node)
    {
        const ISceneNode* tmp = static_cast<const ISceneNode*>(node);

        if(node->isISceneNode())
        {
            if(!tmp->isVisible())
                return false;
        }

        if(node->getParent())
            return isTrulyVisible_static(node->getParent());
        else
            return true;
    }
};

}  // end namespace scene
}  // end namespace nbl

#endif
