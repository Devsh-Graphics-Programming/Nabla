#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		512, 
		512,
		{ 500, 500 },
		{ 1.f, 1.f },
		6,
		5, 
		0.1
	};
	//Sleep(5000);
	WaveSimApp app(params);
	app.Run();
}