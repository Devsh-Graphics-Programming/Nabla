// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_DUMMY_TRANSFORMATION_SCENE_NODE_H_INCLUDED__
#define __NBL_I_DUMMY_TRANSFORMATION_SCENE_NODE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "ISceneNodeAnimator.h"
#include <algorithm>
#include "matrix4x3.h"

namespace nbl
{
namespace scene
{
class ISceneManager;
class ISceneNodeAnimator;
class IDummyTransformationSceneNode;

//! Typedef for array of scene nodes
typedef core::vector<IDummyTransformationSceneNode*> IDummyTransformationSceneNodeArray;
//! Typedef for array of scene node animators
typedef core::vector<ISceneNodeAnimator*> ISceneNodeAnimatorArray;

//! Dummy scene node for adding additional transformations to the scene tree.
/** This scene node does not render itself, and does not respond to set/getPosition,
set/getRotation and set/getScale. Its just a simple scene node that takes a
matrix as relative transformation, making it possible to insert any transformation
anywhere into the scene tree.
This scene node is for example used by the IAnimatedMeshSceneNode for emulating
joint scene nodes when playing skeletal animations.
*/
class IDummyTransformationSceneNode : public virtual core::IReferenceCounted
{
protected:
    uint64_t lastTimeRelativeTransRead[5];

    uint64_t relativeTransChanged;
    bool relativeTransNeedsUpdate;

    virtual ~IDummyTransformationSceneNode()
    {
        removeAll();

        // delete all animators
        ISceneNodeAnimatorArray::iterator ait = Animators.begin();
        for(; ait != Animators.end(); ++ait)
            (*ait)->drop();
    }

public:
    IDummyTransformationSceneNode(IDummyTransformationSceneNode* parent,
        const core::vector3df& position = core::vector3df(0, 0, 0),
        const core::vector3df& rotation = core::vector3df(0, 0, 0),
        const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f))
        : RelativeTranslation(position), RelativeRotation(rotation), RelativeScale(scale),
          Parent(0), relativeTransChanged(1), relativeTransNeedsUpdate(true)
    {
        memset(lastTimeRelativeTransRead, 0, sizeof(uint64_t) * 5);

        if(parent)
            parent->addChild(this);

        updateAbsolutePosition();
    }

    virtual bool isISceneNode() const { return false; }

    //! Returns a reference to the current relative transformation matrix.
    /** This is the matrix, this scene node uses instead of scale, translation
        and rotation. */
    inline virtual const core::matrix4x3& getRelativeTransformationMatrix()
    {
        if(relativeTransNeedsUpdate)
        {
            RelativeTransformation.setRotationDegrees(RelativeRotation);
            RelativeTransformation.setTranslation(RelativeTranslation);
            //
            RelativeTransformation(0, 0) *= RelativeScale.X;
            RelativeTransformation(1, 0) *= RelativeScale.X;
            RelativeTransformation(2, 0) *= RelativeScale.X;
            RelativeTransformation(0, 1) *= RelativeScale.Y;
            RelativeTransformation(1, 1) *= RelativeScale.Y;
            RelativeTransformation(2, 1) *= RelativeScale.Y;
            RelativeTransformation(0, 2) *= RelativeScale.Z;
            RelativeTransformation(1, 2) *= RelativeScale.Z;
            RelativeTransformation(2, 2) *= RelativeScale.Z;
            //
            relativeTransChanged++;
            relativeTransNeedsUpdate = false;
        }
        return RelativeTransformation;
    }

    //!
    inline void setRelativeTransformationMatrix(const core::matrix4x3& tform)
    {
        RelativeTransformation = tform;
        relativeTransChanged++;
        relativeTransNeedsUpdate = false;
    }

    inline const uint64_t& getRelativeTransChangedHint() const { return relativeTransChanged; }

    inline const uint64_t& getAbsoluteTransformLastRecomputeHint() const { return lastTimeRelativeTransRead[3]; }

    inline const core::vector3df& getScale()
    {
        if(lastTimeRelativeTransRead[0] < relativeTransChanged)
        {
            const core::matrix4x3& rel = getRelativeTransformationMatrix();
            RelativeScale = rel.getScale();
            lastTimeRelativeTransRead[0] = relativeTransChanged;
        }
        return RelativeScale;
    }

    inline void setScale(const core::vector3df& scale)
    {
        RelativeScale = scale;
        relativeTransNeedsUpdate = true;
    }

    inline const core::vector3df& getRotation()
    {
        if(lastTimeRelativeTransRead[1] < relativeTransChanged)
        {
            const core::matrix4x3& rel = getRelativeTransformationMatrix();
            RelativeRotation = rel.getRotationDegrees();
            lastTimeRelativeTransRead[1] = relativeTransChanged;
        }
        return RelativeRotation;
    }

    inline void setRotation(const core::vector3df& rotation)
    {
        RelativeRotation = rotation;
        relativeTransNeedsUpdate = true;
    }

    inline const core::vector3df& getPosition()
    {
        if(lastTimeRelativeTransRead[2] < relativeTransChanged)
        {
            const core::matrix4x3& rel = getRelativeTransformationMatrix();
            RelativeTranslation = rel.getTranslation();
            lastTimeRelativeTransRead[2] = relativeTransChanged;
        }
        return RelativeTranslation;
    }

    inline void setPosition(const core::vector3df& newpos)
    {
        RelativeTranslation = newpos;
        relativeTransNeedsUpdate = true;
    }

    inline virtual bool needsAbsoluteTransformRecompute() const
    {
        if(relativeTransNeedsUpdate || lastTimeRelativeTransRead[3] < relativeTransChanged)
            return true;

        if(Parent)
            return lastTimeRelativeTransRead[4] < Parent->getAbsoluteTransformLastRecomputeHint();

        return false;
    }

    inline virtual size_t needsDeepAbsoluteTransformRecompute() const
    {
        const IDummyTransformationSceneNode* parentStack[1024];
        parentStack[0] = this;
        size_t stackSize = 0;

        while(parentStack[stackSize])
        {
            parentStack[++stackSize] = parentStack[stackSize];
            if(stackSize >= 1024)
                break;
        }

        size_t maxStackSize = stackSize - 1;
        while(--stackSize)
        {
            if(parentStack[stackSize]->relativeTransNeedsUpdate || parentStack[stackSize]->lastTimeRelativeTransRead[3] < parentStack[stackSize]->relativeTransChanged)
                return stackSize;

            if(stackSize < maxStackSize)
            {
                if(parentStack[stackSize]->lastTimeRelativeTransRead[4] < parentStack[stackSize + 1]->getAbsoluteTransformLastRecomputeHint())
                    return stackSize;
            }
        }

        return 0xdeadbeefu;
    }

    inline const core::matrix4x3& getAbsoluteTransformation()
    {
        return AbsoluteTransformation;
    }

    //! Gets the absolute position of the node in world coordinates.
    /** If you want the position of the node relative to its parent,
		use getPosition() instead.
		NOTE: For speed reasons the absolute position is not
		automatically recalculated on each change of the relative
		position or by a position change of an parent. Instead the
		update usually happens once per frame in OnAnimate. You can enforce
		an update with updateAbsolutePosition().
		\return The current absolute position of the scene node (updated on last call of updateAbsolutePosition). */
    inline core::vector3df getAbsolutePosition() const
    {
        return AbsoluteTransformation.getTranslation();
    }

    //! Updates the absolute position based on the relative and the parents position
    /** Note: This does not recursively update the parents absolute positions, so if you have a deeper
			hierarchy you might want to update the parents first.*/
    inline virtual void updateAbsolutePosition()
    {
        bool recompute = relativeTransNeedsUpdate || lastTimeRelativeTransRead[3] < relativeTransChanged;

        if(Parent)
        {
            uint64_t parentAbsoluteHint = Parent->getAbsoluteTransformLastRecomputeHint();
            if(lastTimeRelativeTransRead[4] < parentAbsoluteHint)
            {
                lastTimeRelativeTransRead[4] = parentAbsoluteHint;
                recompute = true;
            }

            // recompute if local transform has changed
            if(recompute)
            {
                const core::matrix4x3& rel = getRelativeTransformationMatrix();
                AbsoluteTransformation = concatenateBFollowedByA(Parent->getAbsoluteTransformation(), rel);
                lastTimeRelativeTransRead[3] = relativeTransChanged;
            }
        }
        else if(recompute)
        {
            AbsoluteTransformation = getRelativeTransformationMatrix();
            lastTimeRelativeTransRead[3] = relativeTransChanged;
        }
    }

    //! Returns a const reference to the list of all children.
    /** \return The list of all children of this node. */
    inline const IDummyTransformationSceneNodeArray& getChildren() const
    {
        return Children;
    }

    //! Changes the parent of the scene node.
    /** \param newParent The new parent to be used. */
    virtual void setParent(IDummyTransformationSceneNode* newParent)
    {
        if(newParent == Parent)
            return;

        if(newParent)
        {
            newParent->addChild(this);
            lastTimeRelativeTransRead[4] = 0;
        }
        else
            remove();
    }

    //! Returns the parent of this scene node
    /** \return A pointer to the parent. */
    inline IDummyTransformationSceneNode* getParent() const
    {
        return Parent;
    }

    //! Adds a child to this scene node.
    /** If the scene node already has a parent it is first removed
		from the other parent.
		\param child A pointer to the new child. */
    virtual void addChild(IDummyTransformationSceneNode* child)
    {
        if(!child || child == this || child->getParent() == this)
            return;

        child->grab();
        child->remove();  // remove from old parent
        IDummyTransformationSceneNodeArray::iterator insertionPoint = std::lower_bound(Children.begin(), Children.end(), child);
        Children.insert(insertionPoint, child);
        child->Parent = this;
        child->lastTimeRelativeTransRead[4] = 0;
    }

    //! Removes a child from this scene node.
    /** If found in the children list, the child pointer is also
		dropped and might be deleted if no other grab exists.
		\param child A pointer to the child which shall be removed.
		\return True if the child was removed, and false if not,
		e.g. because it couldn't be found in the children list. */
    virtual bool removeChild(IDummyTransformationSceneNode* child)
    {
        IDummyTransformationSceneNodeArray::iterator found = std::lower_bound(Children.begin(), Children.end(), child);
        if(found == Children.end() || *found != child)
            return false;

        (*found)->Parent = 0;
        (*found)->drop();
        Children.erase(found);
        return true;
    }

    //! Removes all children of this scene node
    /** The scene nodes found in the children list are also dropped
		and might be deleted if no other grab exists on them.
		*/
    virtual void removeAll()
    {
        IDummyTransformationSceneNodeArray::iterator it = Children.begin();
        for(; it != Children.end(); ++it)
        {
            (*it)->Parent = 0;
            (*it)->drop();
        }

        Children.clear();
    }

    //! Removes this scene node from the scene
    /** If no other grab exists for this node, it will be deleted.
		*/
    virtual void remove()
    {
        if(Parent)
            Parent->removeChild(this);
    }

    //! Adds an animator which should animate this node.
    /** \param animator A pointer to the new animator. */
    virtual void addAnimator(ISceneNodeAnimator* animator)
    {
        if(!animator)
            return;

        ISceneNodeAnimatorArray::iterator found = std::lower_bound(Animators.begin(), Animators.end(), animator);
        ///if (found!=Animators.end() && *found==animator) //already in there
        ///return;

        animator->grab();
        Animators.insert(found, animator);
    }

    //! Get a list of all scene node animators.
    /** \return The list of animators attached to this node. */
    inline const ISceneNodeAnimatorArray& getAnimators() const
    {
        return Animators;
    }

    //! Removes an animator from this scene node.
    /** If the animator is found, it is also dropped and might be
		deleted if not other grab exists for it.
		\param animator A pointer to the animator to be deleted. */
    virtual void removeAnimator(ISceneNodeAnimator* animator)
    {
        ISceneNodeAnimatorArray::iterator found = std::lower_bound(Animators.begin(), Animators.end(), animator);
        if(found == Animators.end() || *found != animator)
            return;

        (*found)->drop();
        Animators.erase(found);
        return;
    }

    //! Removes all animators from this scene node.
    /** The animators might also be deleted if no other grab exists
		for them. */
    virtual void removeAnimators()
    {
        ISceneNodeAnimatorArray::iterator it = Animators.begin();
        for(; it != Animators.end(); ++it)
            (*it)->drop();

        Animators.clear();
    }

protected:
    //! Pointer to the parent
    IDummyTransformationSceneNode* Parent;

    //! List of all children of this node
    IDummyTransformationSceneNodeArray Children;

    //! List of all animator nodes
    ISceneNodeAnimatorArray Animators;

    //! Absolute transformation of the node.
    core::matrix4x3 AbsoluteTransformation;

    //! Relative transformation of the node.
    core::matrix4x3 RelativeTransformation;

    //! Relative translation of the scene node.
    core::vector3df RelativeTranslation;

    //! Relative rotation of the scene node.
    core::vector3df RelativeRotation;

    //! Relative scale of the scene node.
    core::vector3df RelativeScale;
};

}  // end namespace scene
}  // end namespace nbl

#endif
