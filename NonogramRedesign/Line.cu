#include "Line.cuh"
#include "Log.h"
#include "device_functions.h"

Line::Line(const Possibilities & possibilities, int lineIndex, bool isRow) :
	possibilities(possibilities), lineIndex(lineIndex), isRow(isRow)
{
	init();
}

Line::Line(const Line& other) :possibilities(other.possibilities),
lineIndex(other.lineIndex), isRow(isRow)
{
	init();
	throw "line copy constractor called";
	LOG(DEBUG) << "line copy constractor\n";
}

Line::~Line()
{
	MemFreeSherd(tempLine);
	MemFreeSherd(validPossibilities);
	//LOG(DEBUG) << "line distractor\n";
}


void __global__ FilterKernal(Cell* line,
	const Cell* possibilities,
	bool* validPossibilities) {	
	
	Cell lineCell = line[threadIdx.x];
	Cell possCell = Possibilities::getLinePtr(possibilities, blockIdx.x, blockDim.x)[threadIdx.x];
	if (lineCell == Cell::WHITE &&
		possCell == Cell::BLACK) {
		validPossibilities[blockIdx.x] = false;
	}
	if (lineCell == Cell::BLACK &&
		possCell == Cell::WHITE) {
		validPossibilities[blockIdx.x] = false;
	}
}

void Line::cpuFilter(const Grid& gridRef) {
	if (isLoaded() == false)
		return;
	Timer::start("cpuFilter");
	gridRef.Read(tempLine, isRow, lineIndex);
	for (int i = 0; i < possibilities.getPossSize(); i++) {
		for (int j = 0; j < possibilities.getLineLen(); j++) {
			Cell c = Cell(tempLine[j] | possibilities.getLinePtr(i)[j]);
			if (c == Cell::BOTH)
				validPossibilities[i] = false;
		}
	}
	Timer::stop("cpuFilter");
}

// filter all the possibilities that isn't valid and return the number of valid possibilities
void Line::Filter(const Grid & gridRef)
{
	if (possibilities.isLoaded() == false)
		return;
	//return cpuFilter(gridRef);
	Timer::start("Filter");
	// copy the line from the grid reference to tempLine
	gridRef.Read(tempLine, isRow, lineIndex);
	gridRef.Read(tempGridRef, isRow, lineIndex); // TODO: debug line
	const int blockSize = possibilities.getPossSize();
	const int threadSize = possibilities.getLineLen();
	const Cell* raw = possibilities.getRawPossibilities();
	FilterKernal<<<blockSize,threadSize>>>
		(tempLine, possibilities.getRawPossibilities(), validPossibilities);
	cudaDeviceSynchronize();

	cudaError err = cudaGetLastError();
	if (err != cudaSuccess) {
		LOG(ERR) << "kernal error: " << err << "\n";
	}
	Timer::stop("Filter");
}

/*__global__ void UnifyKernal(Cell* resLine, const Cell* possibilities, const bool* validPossibilities) {
	Cell lineCell = resLine[threadIdx.x];
	Cell possCell = Possibilities::getLinePtr(possibilities, blockIdx.x, blockDim.x)[threadIdx.x];
	atomicOr((int*)(&resLine[threadIdx.x]),(int) possCell);
}*/
/*

__global__ void UnifyKernal2(const Cell* possibilities, Cell* res, bool* validPossibilities, int firstValid) {
	extern __shared__ Cell sdata[];
	const int tid = threadIdx.x;
	int i = blockIdx.x + threadIdx.x*(gridDim.x);
	sdata[tid] = possibilities[i];
	printf("sdata[%d] = possibilities[%d]. bl.: %d th.: %d\n", tid, i, blockIdx.x, threadIdx.x);
	syncThreads("init");
	if (tid == firstValid && blockDim.x % 2 != 0 && validPossibilities[blockDim.x-1] == true) {
		sdata[firstValid] = Cell(sdata[firstValid] | sdata[blockDim.x - 1]);
		printf("first carry sdata[%d] = sdata[%d] | sdata[%d]. bl.: %d th.: %d\n", firstValid, firstValid,blockDim.x - 1, blockIdx.x, threadIdx.x);
	}
	syncThreads("first carry");

	for (unsigned int s = blockDim.x/2; s > 0; s /= 2) {
		if (tid < s) {
			if (validPossibilities[tid + s] == true) {
				if (validPossibilities[tid] == true) {

				}
			}
			sdata[tid] = Cell(sdata[tid] | sdata[tid + s]);
			printf("sdata[%d] = sdata[%d] | sdata[%d]. bl.: %d th.: %d\n", tid, tid, tid + s, blockIdx.x, threadIdx.x);
		}
		syncThreads("mid for loop");
		if (tid == 0) {
			if (s % 2 != 0 && s != 1) {
				sdata[0] = Cell(sdata[0] | sdata[s - 1]);
				printf("mid carry sdata[0] = sdata[0] | sdata[%d]. bl.: %d th.: %d\n", s - 1, blockIdx.x, threadIdx.x);
			}
		}
		syncThreads("end for loop");
	}
	if (tid == 0) {
		res[blockIdx.x] = sdata[0];
		printf("res[%d] = %d\n", blockIdx.x, res[blockIdx.x]);
	}
}

__global__ void UnifyKernal(const Cell* possibilities, Cell* res) {
	extern __shared__ Cell sdata[];
	const int tid = threadIdx.x;
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	printf("init sdata[%d]= pos[%d]:%d | pos[%d]:%d; blo. %d thr. %d\n",
		 tid, i, possibilities[i], i + blockDim.x, possibilities[i + blockDim.x],
		blockIdx.x, threadIdx.x);
	sdata[tid] = Cell(	possibilities[i] |
						possibilities[i+blockDim.x]);
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
		if (tid < s) {
			printf("sdata[%d]:%d|= sdata[%d]:%d; blo. %d thr. %d\n",
				tid, sdata[tid], tid + s,sdata[tid + s], blockIdx.x, threadIdx.x);
			sdata[tid] = Cell(sdata[tid] | sdata[tid + s]);
		}
		__syncthreads(); 
	}
	if (tid == 0) {
		if (blockDim.x % 2 != 0) {
			printf("end sdata[0] = sdata[0]:%d | sdata[%d]:%d; blo. %d thr. %d\n",
				 sdata[0], blockDim.x - 1,sdata[blockDim.x - 1], blockIdx.x, threadIdx.x);
			sdata[0] = Cell(sdata[0] | sdata[blockDim.x-1]);
		}
		res[blockIdx.x] = sdata[0];
	}
}
*/

__global__ void LargeUnifyKernal(const Cell* possibilities, int* valids, Cell* res, int size) {
	extern __shared__ Cell data[];
	const int elementsPerThread = size / blockDim.x;
	const int startIndex = threadIdx.x * elementsPerThread;
	for (int i = 0; i < elementsPerThread; i++) {
		data[threadIdx.x] =
			Cell(data[threadIdx.x] |
				Possibilities::getLinePtr(possibilities, valids[startIndex + i], gridDim.x)[blockIdx.x]);
	}
	__syncthreads();
	bool carry = blockDim.x % 2;
	//printf("first carry %d. s = %d\n", carry, blockDim.x / 2);
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		if (threadIdx.x < s) {
			//printf("%d) data[%d] = data[%d] | data[%d]\n",blockIdx.x, threadIdx.x, threadIdx.x, threadIdx.x + s);
			data[threadIdx.x] = Cell(data[threadIdx.x] | data[threadIdx.x + s]);
		}
		else if (carry && threadIdx.x == s * 2) {
			//printf("%d) res[%d] = res[%d] | data[%d]\n",blockIdx.x, blockIdx.x, blockIdx.x, threadIdx.x);
			res[blockIdx.x] = Cell(res[blockIdx.x] | data[threadIdx.x]);
		}
		__syncthreads();
		carry = s % 2;
		//printf("new carry: %d, s = %d\n", carry, s/2);
	}
	if (threadIdx.x == 0) {
		res[blockIdx.x] = Cell(res[blockIdx.x] | data[0]);
		//printf("end: res[%d] = res[%d] | data[0]\n", blockIdx.x, blockIdx.x);
	}
}

__global__ void UnifyKernal(const Cell* possibilities, int* valids, Cell* res) {
	extern __shared__ Cell data[];
	//const char sym[3] = { ' ', 'X', '-' };

	/*printf("%d) data[%d] = poss[%d][%d] = %c\n", blockIdx.x, threadIdx.x,
		valids[threadIdx.x], blockIdx.x, sym[Possibilities::getLinePtr(possibilities, valids[threadIdx.x],
			gridDim.x)[blockIdx.x]]);*/
	data[threadIdx.x] = Possibilities::getLinePtr(possibilities, valids[threadIdx.x],
		gridDim.x)[blockIdx.x];
	__syncthreads();
	/*
	if (blockIdx.x == 0) {
		const Cell* l = Possibilities::getLinePtr(possibilities, valids[threadIdx.x], gridDim.x);
		printf("th.%d: poss[%d]\t %c%c%c%c%c%c%c%c%c%c \n", threadIdx.x, valids[threadIdx.x]
			, sym[l[0]], sym[l[1]], sym[l[2]], sym[l[3]], sym[l[4]],
			sym[l[5]], sym[l[6]], sym[l[7]], sym[l[8]], sym[l[9]]);
	}
	__syncthreads();*/

	
	bool carry = blockDim.x % 2;
	//printf("first carry %d. s = %d\n", carry, blockDim.x / 2);
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		if (threadIdx.x < s) {
			//printf("%d) data[%d] = data[%d] | data[%d]\n",blockIdx.x, threadIdx.x, threadIdx.x, threadIdx.x + s);
			data[threadIdx.x] = Cell(data[threadIdx.x] | data[threadIdx.x + s]);
		}
		else if (carry && threadIdx.x == s * 2) {
			//printf("%d) res[%d] = res[%d] | data[%d]\n",blockIdx.x, blockIdx.x, blockIdx.x, threadIdx.x);
			res[blockIdx.x] = Cell(res[blockIdx.x] | data[threadIdx.x]);
		}
		__syncthreads();
		carry = s % 2;
		//printf("new carry: %d, s = %d\n", carry, s/2);
	}
	if (threadIdx.x == 0) {
		res[blockIdx.x] = Cell(res[blockIdx.x] | data[0]);
		//printf("end: res[%d] = res[%d] | data[0]\n", blockIdx.x, blockIdx.x);
	}
}
Cell* Line::cpuUnify() {
	Timer::start("cpuUnify");
	cudaMemset(tempLine, Cell::UNKNOWN, sizeof(Cell)*possibilities.getLineLen());
	for (int i = 0; i < possibilities.getPossSize(); i++) {
		if (validPossibilities[i] == true) {
			for (int j = 0; j < possibilities.getLineLen(); j++) {
				tempLine[j] = Cell(tempLine[j] | possibilities.getLinePtr(i)[j]);
			}
		}
	}
	Timer::stop("cpuUnify");
	return tempLine;
}




Cell* Line::Unify()
{
	//return cpuUnify();
	Timer::start("Unify");
	const int numOfPoss = possibilities.getPossSize();
	const int lineLen = possibilities.getLineLen();
	const Cell* raw = possibilities.getRawPossibilities();

	validPossIndexs.clear();
	for (int i = 0; i < numOfPoss; i++) {
		if (validPossibilities[i] == true) {
			validPossIndexs.push_back(i);
		}
	}
	if (validPossIndexs.Size() == 0) {
		Timer::abort("Unify");
		return NULL;
	}

	cudaMemset(tempLine, Cell::UNKNOWN, sizeof(Cell)*lineLen);
	int threads = 1023;
	int blocks = lineLen;
	int mem;
	//printf("block size: %d thread size: %d\n", lineLen, validPossIndexs.Size());
	if (validPossIndexs.Size() <= 1023) {
		Timer::start("UnifyKernal");
		threads = validPossIndexs.Size();
		mem = sizeof(Cell)*threads;
		UnifyKernal << <blocks, threads, mem >> >
			(raw, validPossIndexs.getRaw(), tempLine);
		cudaDeviceSynchronize();
		Timer::stop("UnifyKernal");
	}
	else {
		Timer::start("LargeUnifyKernal");
		
		while (validPossIndexs.Size() % threads)
			threads--;
		mem = sizeof(Cell)*threads;
		LargeUnifyKernal << <blocks, threads,  mem>> >
			(possibilities.getRawPossibilities(), 
				validPossIndexs.getRaw(), tempLine, validPossIndexs.Size());
		cudaDeviceSynchronize();
		Timer::stop("LargeUnifyKernal");
	}
	
	cudaError err = cudaGetLastError();
	if (err != cudaSuccess) {
		LOG(ERR) << "cuda error at unify: " << cudaGetErrorName(err) << " line: " << toString() << "\n";
		printf("bl.:%d th.:%d mem.:%d\n", blocks, threads, mem);
		system("pause");
	}
	//cpuUnify();
	Timer::stop("Unify");
	/*
	Cell* temp = new Cell[lineLen * sizeof(Cell)];
	cudaMemcpy(temp, tempLine, sizeof(Cell)*lineLen, cudaMemcpyDeviceToHost);
	cpuUnify();
	for (int i = 0; i < lineLen; i++) {
		if (temp[i] != tempLine[i]) {
			char syms[4] = { '_', 'X', '.', '?' };
			printf("error, cup unify and kernal unify unequal\n");
			printf("\nOptions:\n");
			for (int j = 0; j < possibilities.getPossSize(); j++) {
				if (validPossibilities[j] == true) {
					printf("     ");
					for (int k = 0; k < possibilities.getLineLen(); k++) {
						printf("%c", syms[possibilities.getLinePtr(j)[k]]);
					}
					printf("\n");
				}
			}

			printf("cpu: ");
			for (int j = 0; j < lineLen; j++) {
				printf("%c", syms[tempLine[j]]);
			}
			printf("\ngpu: ");
			for (int j = 0; j < lineLen; j++) {
				printf("%c", syms[temp[j]]);
			}
			printf("\nori: ");
			for (int j = 0; j < lineLen; j++) {
				printf("%c", syms[tempGridRef[j]]);
			}
			printf("\n");
			printf("poss size: %d valid poss size: %d\n", possibilities.getPossSize(), validPossIndexs.Size());

			system("pause");
			break;
		}
	}
	free(temp);*/
	return tempLine;
}

bool Line::isLoaded() const
{
	return possibilities.isLoaded();
}

int Line::getLineLen() const
{
	return possibilities.getLineLen();
}

int Line::getLineIndex() const
{
	return lineIndex;
}

void Line::resetValids()
{
	cudaMemset(validPossibilities, true, possibilities.getPossSize() * sizeof(bool));
}

std::string Line::toString() const
{
	std::string ret = isRow ? "row " : "col ";
	ret += std::to_string(lineIndex) + " " + std::to_string(possibilities.getPossSize());
	return ret;
}

void Line::init()
{
	MemAllocSherd(&tempLine, sizeof(Cell)*possibilities.getLineLen());
	MemAllocSherd(&validPossibilities, sizeof(bool) * possibilities.getPossSize());
	MemAllocSherd(&tempGridRef, sizeof(Cell) * possibilities.getLineLen());// TODO: debug line;
	cudaMemset(validPossibilities, true, sizeof(bool) * possibilities.getPossSize());
	validPossIndexs.setCapacity(possibilities.getPossSize());
}



