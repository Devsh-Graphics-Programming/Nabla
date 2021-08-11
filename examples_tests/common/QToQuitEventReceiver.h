// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_Q_TO_QUIT_EVENT_RECEIVER_H__INCLUDED__
#define __NBL_Q_TO_QUIT_EVENT_RECEIVER_H__INCLUDED__

#include <nabla.h>

using namespace nbl;
using namespace ui;

/*
	Simple event receiver for most examples
	that closes the engine when Q is pressed
*/

class QToQuitEventReceiver
{
	public:
		QToQuitEventReceiver() : running(true) {}
		virtual ~QToQuitEventReceiver() {}

		void process(const IKeyboardEventChannel::range_t& events)
		{
			for (auto eventIterator = events.begin(); eventIterator != events.end(); eventIterator++)
			{
				auto event = *eventIterator;

				if (event.keyCode == nbl::ui::EKC_Q)
					running = false;
			}
		}

		inline bool keepOpen() const { return running; }

	private:
		bool running;
};

#endif // __NBL_Q_TO_QUIT_EVENT_RECEIVER_H__INCLUDED__