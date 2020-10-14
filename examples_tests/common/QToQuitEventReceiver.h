// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_Q_TO_QUIT_EVENT_RECEIVER_H__INCLUDED__
#define __NBL_Q_TO_QUIT_EVENT_RECEIVER_H__INCLUDED__

#include "irrlicht.h"

//! Simple event receiver for most examples that closes the engine when Q is pressed
class QToQuitEventReceiver : public irr::IEventReceiver
{
	public:
		QToQuitEventReceiver() : running(true)
		{
		}

		bool OnEvent(const irr::SEvent& event)
		{
			if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
			{
				switch (event.KeyInput.Key)
				{
					case irr::KEY_KEY_Q: // switch wire frame mode
						running = false;
						return true;
					default:
						break;
				}
			}

			return false;
		}

		inline bool keepOpen() const { return running; }

	private:
		bool running;
};

#endif