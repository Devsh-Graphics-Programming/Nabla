#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		256, 
		256,
		{ 200.f, 200.f },
		{ 1.f, 1.f },
		8,
		8, 
		0.1
	};
	//Sleep(5000);
	WaveSimApp app(params);
	app.Run();
}