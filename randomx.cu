//
//  RandomX algo (with smart contracts header)
//  Based on tpruvot's implementation from May 2018
//
//  Based on the work of SChernykh:	https://github.com/SChernykh/RandomX_CUDA/tree/master/RandomX_CUDA
//			 tevador:	https://github.com/tevador/RandomX/
//			 Barrystyle
//
//  LUXcore dev team - 2020
//
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include "uint256.h"


#include "RandomX/src/blake2/blake2.h"
#include "RandomX/src/aes_hash.hpp"
#include "RandomX/src/randomx.h"
#include "RandomX/src/configuration.h"
#include "RandomX/src/common.hpp"

extern "C" {
#include <rxhash/rx-slow-hash.h>
}

#include "randomX_CUDA/blake2b_cuda.hpp"
#include "randomX_CUDA/aes_cuda.hpp"
#include "randomX_CUDA/randomx_cuda.hpp"

#include "miner.h"
#include "cuda_helper.h"

static bool init[MAX_GPUS] = { 0 };

static void* dataset_gpu[MAX_GPUS];
static void* scratchpads_gpu[MAX_GPUS];
static void* hashes_gpu[MAX_GPUS];
static void* entropy_gpu[MAX_GPUS];
static void* vm_states_gpu[MAX_GPUS];
static void* rounding_gpu[MAX_GPUS];
static void* num_vm_cycles_gpu[MAX_GPUS];
static uint64_t* blockTemplate_gpu[MAX_GPUS];
static uint32_t* d_resNonces[MAX_GPUS];

static uint256 ulocalseed[MAX_GPUS];
static uint32_t batch_size[MAX_GPUS];
static size_t dataset_size;
static randomx_cache *randomx_cpu_cache;
static randomx_dataset *randomx_cpu_dataset;

static struct Dataset_threads {
	pthread_t thr;
	int id;
} *dataset_threads;

static pthread_barrier_t mybarrier;

__constant__ uint32_t gpu_target[8]; // 32 bytes

#define maxResults 2

extern "C" void randomx_hash(void *output, const void *input){}

extern "C" void randomx_init_barrier(const int num_threads){
	pthread_barrier_init(&mybarrier, NULL, num_threads);
}

static void* dataset_init_cpu_thr(void *arg){

	int i = *(int*)arg;
	uint32_t n = num_cpus;
	randomx_init_dataset(randomx_cpu_dataset, randomx_cpu_cache, (i * randomx_dataset_item_count()) / n, ((i + 1) * randomx_dataset_item_count()) / n - (i * randomx_dataset_item_count()) / n);
	return NULL;
}
static void init_dataset_gpu(const int thr_id, const char* mySeed){

	if(thr_id==0){
		struct timeval tv_start,tv_end,tv_diff;

		gettimeofday(&tv_start, NULL);
		gpulog(LOG_INFO,thr_id,"Initializing dataset...");
		randomx_cpu_cache = randomx_alloc_cache((randomx_flags)(RANDOMX_FLAG_JIT));
		randomx_init_cache(randomx_cpu_cache, mySeed, 64);

		randomx_cpu_dataset =randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
		if (!randomx_cpu_dataset){
			gpulog(LOG_WARNING,thr_id,"Couldn't allocate dataset using large pages");
		}

		applog(LOG_NOTICE,"Initializing dataset using %u cpu cores",num_cpus);

		if(dataset_threads==NULL){
			dataset_threads = (Dataset_threads*)malloc(num_cpus*sizeof(Dataset_threads));
		}

		for (int i = 0; i < num_cpus; ++i){
			dataset_threads[i].id = i;
			pthread_create(&dataset_threads[i].thr,NULL,dataset_init_cpu_thr,(void*)&dataset_threads[i].id);
		}
		for (int i = 0; i < num_cpus; ++i){
			pthread_join(dataset_threads[i].thr,NULL);
		}

		randomx_release_cache(randomx_cpu_cache);

		CUDA_SAFE_CALL(cudaMemcpy(dataset_gpu[thr_id], randomx_get_dataset_memory(randomx_cpu_dataset), dataset_size, cudaMemcpyHostToDevice));
		randomx_release_dataset(randomx_cpu_dataset);
		gettimeofday(&tv_end, NULL);
		timeval_subtract(&tv_diff, &tv_end, &tv_start);
		double dtime = (double) tv_diff.tv_sec*1e3 + 1e-3 * tv_diff.tv_usec;
		gpulog(LOG_INFO,thr_id,"done in %.3f ms\n", dtime);
		pthread_barrier_wait(&mybarrier);
	}else{
		pthread_barrier_wait(&mybarrier);
		CUDA_SAFE_CALL(cudaMemcpy(dataset_gpu[thr_id], dataset_gpu[0], dataset_size, cudaMemcpyDeviceToDevice));
	}
}

__device__ __forceinline__
static bool hashbelowtarget(const uint32_t *const __restrict__ hash, const uint32_t *const __restrict__ target)
{
	#pragma unroll 8
	for(int i=7;i>=0;i--){
		if (hash[i] > target[i])
			return false;
		if (hash[i] < target[i])
			return true;
	}
	return true;
}

__global__ __launch_bounds__(256, 1)
static void checkhash_32(uint32_t threads, uint32_t startNounce, uint32_t *hash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *inpHash = &hash[thread << 3];
		if (hashbelowtarget(inpHash, gpu_target)){
			uint32_t pos = atomicInc(&resNonces[0],0xffffffff)+1;
			if(pos <= maxResults)//return 3 vals. 0->num of nonces 1,2 the nonces
				resNonces[pos] = (startNounce + thread);
		}
	}
}

static void check_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash, uint32_t *h_ret)
{
	cudaMemset(d_resNonces[thr_id], 0x0, sizeof(uint32_t));

	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	checkhash_32 <<<grid, block>>> (threads, startNounce, d_inputHash, d_resNonces[thr_id]);
	cudaDeviceSynchronize();
	cudaMemcpy(h_ret, d_resNonces[thr_id], (maxResults+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return;
}

extern "C" int scanhash_randomx(int thr_id, struct work* work, const uint8_t* seedhash,uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, dev_id);
	uint32_t throughput;

	switch (device_sm[dev_id]) 
	{
		case 520: // not sure there are card with enough memory
		case 610:
			throughput = (props.multiProcessorCount) * 48;
			break;
		case 700:
			throughput = 2752; //Titan
			break;
		case 750:
			throughput = 0; // use (freemem/props.multiProcessorCount)*props.multiProcessorCount
			break;
		case 800:
			throughput = 0;
			break;
		case 860:
			throughput = (props.multiProcessorCount) * 64;
			break;
		default:
			throughput = 0;
	}

	throughput = cuda_default_throughput(thr_id, throughput);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);
//	throughput = 0;
	uint256 seedhashA;
	memcpy(&seedhashA,seedhash,32);

	if (opt_benchmark){
		memset(ptarget,0xFF,8*sizeof(uint32_t));
		ptarget[7] = 0x000FFFFF;
	}

	if (!init[thr_id]){
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}

		size_t free_mem, total_mem;
		CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
		gpulog(LOG_INFO, thr_id, "%zu/%zu MB GPU memory free", free_mem >> 20, total_mem >> 20);

		// There should be enough GPU memory for the 6080 MB dataset, 32 scratchpads and 64 MB for everything else
		dataset_size = (uint64_t)randomx_dataset_item_count() * (uint64_t)RANDOMX_DATASET_ITEM_SIZE;
		if (free_mem <= dataset_size + (32U * (RANDOMX_SCRATCHPAD_L3 + 64)) + (64U << 20)){
			gpulog(LOG_INFO, thr_id, "Not enough free GPU memory!");
			return false;
		}

		// prevent ccminer crashing because of throughput exceeding memory capacity
		uint32_t available_gram = ((free_mem - (dataset_size)-(64U << 20)) / (RANDOMX_SCRATCHPAD_L3 + 64));
		uint32_t refactored_thr = (available_gram/props.multiProcessorCount) * props.multiProcessorCount ;
		
		batch_size[thr_id] = (throughput > available_gram || throughput == 0) ? refactored_thr : throughput;
//		batch_size[thr_id] = (batch_size[thr_id] / 32) * 32; 

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads processor count %u device_sm %u ", throughput2intensity(batch_size[thr_id]), batch_size[thr_id], props.multiProcessorCount, device_sm[dev_id]);

		CUDA_SAFE_CALL(cudaMalloc(&dataset_gpu[thr_id], dataset_size));
		gpulog(LOG_INFO, thr_id, "Allocated %.0f MB dataset", dataset_size / 1048576.0);

		CUDA_SAFE_CALL(cudaMalloc(&scratchpads_gpu[thr_id], (size_t)batch_size[thr_id] * (RANDOMX_SCRATCHPAD_L3 + 64)));
		if (!scratchpads_gpu[thr_id]){
			gpulog(LOG_ERR,thr_id, "Failed to allocate GPU memory for scratchpads!");
			return false;
		}else{
			gpulog(LOG_INFO,thr_id,"Allocated %u scratchpads", batch_size[thr_id]);
		}
		CUDA_SAFE_CALL(cudaMalloc(&hashes_gpu[thr_id], (size_t)batch_size[thr_id] * HASH_SIZE));
		if (!hashes_gpu[thr_id]){
			gpulog(LOG_ERR,thr_id, "Failed to allocate GPU memory for hashes!");
			return false;
		}
		CUDA_SAFE_CALL(cudaMalloc(&entropy_gpu[thr_id], (size_t)batch_size[thr_id] * ENTROPY_SIZE));
		if (!entropy_gpu[thr_id]){
			gpulog(LOG_ERR,thr_id, "Failed to allocate GPU memory for entropy!");
			return false;
		}
		CUDA_SAFE_CALL(cudaMalloc(&vm_states_gpu[thr_id], (size_t)batch_size[thr_id] * RANDOMX_PROGRAM_SIZE * sizeof(uint64_t)));
		if (!vm_states_gpu[thr_id]){
			gpulog(LOG_ERR,thr_id, "Failed to allocate GPU memory for hashes!");
			return false;
		}
		CUDA_SAFE_CALL(cudaMalloc(&rounding_gpu[thr_id], (size_t)batch_size[thr_id] * sizeof(uint32_t)));
		if (!rounding_gpu[thr_id]){
			gpulog(LOG_ERR,thr_id, "Failed to allocate GPU memory for rounding!");
			return false;
		}
		CUDA_SAFE_CALL(cudaMalloc(&num_vm_cycles_gpu[thr_id], sizeof(uint64_t)));
		if (!num_vm_cycles_gpu[thr_id]){
			gpulog(LOG_ERR,thr_id, "Failed to allocate GPU memory for VM cycles!");
			return false;
		}
		cudaMemset(num_vm_cycles_gpu[thr_id], 0, sizeof(uint64_t));

		CUDA_SAFE_CALL(cudaMalloc(&blockTemplate_gpu[thr_id], 144*sizeof(uint8_t)));
		if (!blockTemplate_gpu[thr_id]){
			gpulog(LOG_ERR,thr_id, "Failed to allocate GPU memory for block header!");
			return false;
		}

		CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], (maxResults+1)*sizeof(uint32_t)));		
		init[thr_id] = true;

		CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
		gpulog(LOG_INFO,thr_id,"%zu MB free GPU memory left\n", free_mem >> 20);

		init_dataset_gpu(thr_id,(const char*)seedhashA.GetHex().c_str());

		ulocalseed[thr_id]=seedhashA;
	}


	if (ulocalseed[thr_id]!=seedhashA) {
		gpulog(LOG_WARNING,thr_id,"New seed detected");
		ulocalseed[thr_id]=seedhashA;
		init_dataset_gpu(thr_id,(const char*)ulocalseed[thr_id].GetHex().c_str());
	}
	uint32_t endiandata[36];
	for (int k = 0; k < 36; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}
	CUDA_SAFE_CALL(cudaMemcpy(blockTemplate_gpu[thr_id], endiandata, 36*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(gpu_target, ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	do{
		blake2b_initial_hash_144<<<batch_size[thr_id] / 32, 32>>>(hashes_gpu[thr_id], blockTemplate_gpu[thr_id], pdata[19]);
		//initScratchpad
		fillAes1Rx4<RANDOMX_SCRATCHPAD_L3, false, 64><<<batch_size[thr_id] / 32, 32 * 4>>>(hashes_gpu[thr_id], scratchpads_gpu[thr_id], batch_size[thr_id]);
		//reset rounding mode
		CUDA_SAFE_CALL(cudaMemset(rounding_gpu[thr_id], 0, batch_size[thr_id] * sizeof(uint32_t)));

		for (size_t i = 0; i < RANDOMX_PROGRAM_COUNT; ++i){
			fillAes4Rx4<ENTROPY_SIZE, false><<<batch_size[thr_id] / 32, 32 * 4>>>(hashes_gpu[thr_id], entropy_gpu[thr_id], batch_size[thr_id]);
			init_vm<<<batch_size[thr_id] / 4, 32>>>(entropy_gpu[thr_id], vm_states_gpu[thr_id], num_vm_cycles_gpu[thr_id]);
			execute_vm<<<batch_size[thr_id] / 2, 16>>>((uint64_t*)vm_states_gpu[thr_id], rounding_gpu[thr_id], (uint64_t*)scratchpads_gpu[thr_id], dataset_gpu[thr_id]);
			if (i == RANDOMX_PROGRAM_COUNT - 1){
				hashAes1Rx4<RANDOMX_SCRATCHPAD_L3, 192, RANDOMX_PROGRAM_SIZE * sizeof(uint64_t), 64><<<batch_size[thr_id] / 32, 32 * 4>>>(scratchpads_gpu[thr_id], vm_states_gpu[thr_id], batch_size[thr_id]);
				blake2b_hash_registers<REGISTERS_SIZE, RANDOMX_PROGRAM_SIZE * sizeof(uint64_t), 32><<<batch_size[thr_id] / 32, 32>>>(hashes_gpu[thr_id], vm_states_gpu[thr_id]);
			}else{
				blake2b_hash_registers<REGISTERS_SIZE, RANDOMX_PROGRAM_SIZE * sizeof(uint64_t), 64><<<batch_size[thr_id] / 32, 32>>>(hashes_gpu[thr_id], vm_states_gpu[thr_id]);
			}
		}

		uint32_t h_resNonces[maxResults+1];
		check_hash_32(thr_id,batch_size[thr_id],pdata[19], (uint32_t*)hashes_gpu[thr_id], h_resNonces);

		if (h_resNonces[0]>0) {
			const uint64_t Htarg = ((uint64_t*)ptarget)[3];
			uint32_t vhash[8];
			be32enc(&endiandata[19], h_resNonces[1]);
			
		        rx_slow_hash(0, 0, (const char*)seedhashA.GetHex().c_str(), (const char*)endiandata, 144, (char*)vhash, 0, 0);
			if (((uint64_t*)vhash)[3] <= Htarg /*&& fulltest(vhash, ptarget)*/){
				work->valid_nonces = 1;
				work->nonces[0] = h_resNonces[1];
				work_set_target_ratio(work, vhash);
				*hashes_done = pdata[19] - first_nonce + batch_size[thr_id];

				if(h_resNonces[0]>1){
					work->nonces[1] = h_resNonces[2];
					be32enc(&endiandata[19], work->nonces[1]);
				        rx_slow_hash(0, 0, (const char*)seedhashA.GetHex().c_str(), (const char*)endiandata, 144, (char*)vhash, 0, 0);
					if (((uint64_t*)vhash)[3] <= Htarg /*&& fulltest(vhash, ptarget)*/){
						bn_set_target_ratio(work, vhash, 1);
						work->valid_nonces++;
						pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
					}else{
						if (!opt_quiet)
							gpulog(LOG_WARNING, thr_id, "supp result for %u does not validate on CPU! throughput=%u", work->nonces[1], batch_size);
					}
				}else{				
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				if (pdata[19] > max_nonce) pdata[19] = max_nonce;
				return work->valid_nonces;
			} else if (vhash[7] > Htarg){
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %u does not validate on CPU! throughput=%u", work->nonces[0], batch_size);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t)batch_size[thr_id] + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += batch_size[thr_id];

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_randomx(int thr_id)
{
	if (!init[thr_id])
		return;

	if(thr_id==0){
		free(dataset_threads);
	}
	cudaDeviceSynchronize();
	cudaFree(dataset_gpu[thr_id]);
	cudaFree(scratchpads_gpu[thr_id]);
	cudaFree(hashes_gpu[thr_id]);
	cudaFree(entropy_gpu[thr_id]);
	cudaFree(vm_states_gpu[thr_id]);
	cudaFree(rounding_gpu[thr_id]);
	cudaFree(num_vm_cycles_gpu[thr_id]);
	cudaFree(blockTemplate_gpu[thr_id]);
	cudaFree(d_resNonces[thr_id]);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
