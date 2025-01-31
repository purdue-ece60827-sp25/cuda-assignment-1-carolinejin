
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here

	int i = blockIdx.x*blockDim.x + threadIdx.x;
  	if (i < size) y[i] = scale*x[i] + y[i];

}

int runGpuSaxpy(int vectorSize) {
	
	srand(time(0));

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	//std::cout << "Lazy, you are!\n";
	///std::cout << "Write code, you must\n";

	float *x, *y, *c, *d_x, *d_y;

	x = (float *) malloc(vectorSize * sizeof(float));
	y = (float *) malloc(vectorSize * sizeof(float));
	c = (float *) malloc(vectorSize * sizeof(float));

	cudaMalloc(&d_x, vectorSize*sizeof(float)); 
	cudaMalloc(&d_y, vectorSize*sizeof(float));


	for (int idx = 0; idx < vectorSize; ++idx) {
		x[idx] = (float)(rand()) / (float)(rand());
		y[idx] = (float)(rand()) / (float)(rand());
	}

	std::memcpy(c, y, vectorSize * sizeof(float));

	cudaMemcpy(d_x, x, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
	
	float scale = (float)(rand()) / (float)(rand());

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" x = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%f, ", x[i]);
		}
		printf(" ... }\n");
		printf(" y = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%f, ", y[i]);
		}
		printf(" ... }\n");
	#endif

	saxpy_gpu<<<ceil(vectorSize/256.0),256>>>(d_x, d_y, scale, vectorSize);
	cudaMemcpy(y, d_y, vectorSize*sizeof(float), cudaMemcpyDeviceToHost);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" y = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%f, ", y[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(x, c, y, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";


	cudaFree(d_x);
  	cudaFree(d_y);
  	free(x);
  	free(y);


	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	//int shared_blocks[500];

	int threadId = blockIdx.x*blockDim.x + threadIdx.x;

	uint64_t hitCount = 0;
	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), threadId, 0, &rng);
	//printf("thread id: " + threadId + '\n');
    // Get a new random value
	if (threadId < pSumSize){
		for (int i = 0; i < sampleSize; ++i) 
		{        
			float x = curand_uniform(&rng);       
			float y = curand_uniform(&rng);

			if (int(x * x + y * y) == 0){
				hitCount++;
			}
			
		}
		pSums[threadId]=hitCount;
	}
	

	/*shared_blocks[threadIdx.x] = hitCount;


	if (threadIdx.x == 0){
		int total = 0;
		for (int j = 0; j < pSumSize; j++){
			total += shared_blocks[j];
		}
		pSums[blockIdx.x] = total;
	}*/
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here

	int blocks = std::ceil(generateThreadCount/256.0);
	uint64_t *host_count, *device_count;
	//host_count = new uint64_t[generateThreadCount];
	host_count = (uint64_t *) malloc(generateThreadCount * sizeof(uint64_t));
	cudaMalloc(&device_count, sizeof(uint64_t)*generateThreadCount);

	generatePoints<<<blocks, 256>>>(device_count, generateThreadCount, sampleSize);

	cudaMemcpy(host_count, device_count, sizeof(uint64_t)*generateThreadCount, cudaMemcpyDeviceToHost);
	cudaFree(device_count);

	uint64_t total = 0;

	for (int i = 0; i < generateThreadCount; i++) {
		std::cout<<"Thread #"<<i<<" count: "<<host_count[i]<<"\n";
		total += host_count[i];
		std::cout<<"total"<<total<<"\n";
	}
	uint64_t tests = generateThreadCount* sampleSize;
	std::cout<<"total tests "<<tests<<"\n";
	approxPi = 4.0 * (double)total/(double)tests;
	
	return approxPi;


}
