#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		256, 
		256,
		{ 1000, 1000 },
		{ 1.f, 0.f },
		20,
		0.0001, 
		0.
	};
	//Sleep(5000);
	WaveSimApp app(params);
	app.Run();
}