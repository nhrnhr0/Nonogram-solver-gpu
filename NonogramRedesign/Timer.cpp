#include "Timer.h"
#include <math.h>
#include <algorithm>
#include <assert.h>

std::map<std::string, timeData> Timer::times = [] {
	std::map < std::string, timeData> ret;
	return ret;
}();


const timeData & Timer::start(const std::string& name)
{
	auto& timer = times[name];
	// make sure the timer isn't running
	assert(timer.isRunning == false);
	timer.isRunning = true;
	timer.curr = std::clock();
	timer.count++;
	return timer;
}
void Timer::abort(const std::string& name) {
	auto& timer = times[name];
	timer.isRunning = false;
	timer.count--;
}
const timeData & Timer::stop(const std::string& name)
{
	auto& timer = times[name];
	// make sure the timer is running
	assert(timer.isRunning == true);
	timer.isRunning = false;
	double currTime = double(std::clock() - timer.curr) / CLOCKS_PER_SEC;
	if (timer.count == 0)
		throw std::exception((std::string("didn't find timer with this name ") + name).c_str());

	else if (timer.count == 1) {
		timer.max = currTime;
		timer.min = currTime;
	}
	timer.max = std::max(timer.max, currTime);
	timer.min = std::min(timer.min, currTime);
	timer.sum += currTime;

	timer.curr = 0;
	return timer;
}

void Timer::print(std::string name)
{
	auto& timer = times[name];
	print(std::pair<std::string, timeData>(name,timer));
}

void Timer::print(std::pair<std::string, timeData> data) {
	printf("%s)\targ: %lfs\tmax: %lfs\tmin: %lfs\n", data.first.c_str(),
		data.second.sum / data.second.count,
		data.second.max, data.second.min);
}

void Timer::printAll()
{
	printf("\n");
	for (auto i : times) {
		print(i);
	}
}

Timer::Timer()
{
}


Timer::~Timer()
{
}
