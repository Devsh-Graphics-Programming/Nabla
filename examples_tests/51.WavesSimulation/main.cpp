#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		128, 
		128,
		{ 1.f, 1.f },
		{ 1.f, 1.f },
		6,
		25, 
		0
	};
	WaveSimApp app(params);
	app.Run();
}