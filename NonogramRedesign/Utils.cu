#include "Utils.cuh"


Grid::Grid(const Grid & other)
{
	Setup(other.heigth, other.width);
	Set(other);
}

Grid & Grid::operator=(const Grid & other)
{
	Setup(other.heigth, other.width);
	Set(other);
	return *this;
}

Grid::Grid(int height, int width) {
	Setup(height, width);
}

void Grid::Set(const Grid & other)
{
	cudaMemcpy(data, other.data, sizeof(Cell) * Size(), cudaMemcpyDeviceToDevice);
}

Grid::~Grid()
{
	MemFreeSherd(data);
}

void Grid::Read(Cell * buffer, bool isRow, int index) const
{
	isRow ? ReadRow(buffer, index) : ReadCol(buffer, index);
}

void Grid::ReadRow(Cell * buffer, int index) const
{
	cudaMemcpy(buffer, this->at(index), sizeof(Cell) * width, cudaMemcpyDeviceToDevice);
}

void Grid::ReadCol(Cell * buffer, int index) const
{
	for (int i = 0; i < heigth; i++) {
		buffer[i] = this->at(i, index);
	}
}

int Grid::Size() const
{
	return heigth*width;
}

void Grid::Setup(int heigth, int width)
{
	this->heigth = heigth;
	this->width = width;
	MemAllocSherd(&data, sizeof(Cell) * heigth * width);
}

const Cell & Grid::at(int row, int col) const
{
	assert(row < heigth);
	assert(col < width);
	return data[row * width + col];
}

const Cell * Grid::at(int row) const
{
	return data + (row * width);
}

Cell & Grid::at(int row, int col) {
	assert(row < heigth);
	assert(col < width);
	return data[row * width + col];
}

Cell * Grid::at(int row) {
	return data + (row * width);
}



int cmd_c(char *cmd, char *output, DWORD maxbuffer)
{
	HANDLE readHandle;
	HANDLE writeHandle;
	HANDLE stdHandle;
	DWORD bytesRead;
	DWORD retCode;
	SECURITY_ATTRIBUTES sa;
	PROCESS_INFORMATION pi;
	STARTUPINFO si;

	ZeroMemory(&sa, sizeof(SECURITY_ATTRIBUTES));
	ZeroMemory(&pi, sizeof(PROCESS_INFORMATION));
	ZeroMemory(&si, sizeof(STARTUPINFO));

	sa.bInheritHandle = true;
	sa.lpSecurityDescriptor = NULL;
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	si.cb = sizeof(STARTUPINFO);
	si.dwFlags = STARTF_USESHOWWINDOW;
	si.wShowWindow = SW_HIDE;

	if (!CreatePipe(&readHandle, &writeHandle, &sa, NULL))
	{
		OutputDebugString("cmd: CreatePipe failed!\n");
		return 0;
	}

	stdHandle = GetStdHandle(STD_OUTPUT_HANDLE);

	if (!SetStdHandle(STD_OUTPUT_HANDLE, writeHandle))
	{
		OutputDebugString("cmd: SetStdHandle(writeHandle) failed!\n");
		return 0;
	}

	if (!CreateProcess(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi))
	{
		OutputDebugString("cmd: CreateProcess failed!\n");
		return 0;
	}

	GetExitCodeProcess(pi.hProcess, &retCode);
	while (retCode == STILL_ACTIVE)
	{
		GetExitCodeProcess(pi.hProcess, &retCode);
	}

	if (!ReadFile(readHandle, output, maxbuffer, &bytesRead, NULL))
	{
		OutputDebugString("cmd: ReadFile failed!\n");
		return 0;
	}
	output[bytesRead] = '\0';

	if (!SetStdHandle(STD_OUTPUT_HANDLE, stdHandle))
	{
		OutputDebugString("cmd: SetStdHandle(stdHandle) failed!\n");
		return 0;
	}

	if (!CloseHandle(readHandle))
	{
		OutputDebugString("cmd: CloseHandle(readHandle) failed!\n");
	}
	if (!CloseHandle(writeHandle))
	{
		OutputDebugString("cmd: CloseHandle(writeHandle) failed!\n");
	}

	return 1;
}

std::string cmd(char* cmd) {
	const int outputSize = 2048;
	char output[outputSize];
	cmd_c(cmd, output, outputSize);
	return std::string(output);
}

void printLine(const Cell* line, const int len) {
	const char syms[] = { ' ', 'X', '_', '?' };
	for (int i = 0; i < len; i++) {
		std::cout << syms[line[i]];
	}
	std::cout << std::endl;
}