#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		1024, 
		1024,
		{ 1.f, 1.f },
		{ 1.f, 1.f },
		50,
		1
	};
	//Sleep(5000);
	WaveSimApp app(params);
	app.Run();
}