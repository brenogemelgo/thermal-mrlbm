#if defined(CASE_JET)
#include "boundary/hostInitialization.cuh"
#endif
#include "initialConditions.cuh"
#include "kernel.cuh"
#include "checkpoint.cuh"

#define CUDA_CHECK(call) outputCheckCuda((call),#call)

static bool validRunId(const std::string &s)
{
    return !s.empty() && s!="." && s!=".." && s.find('/')==s.npos && s.find('\\')==s.npos;
}
static natural_t parseStep(const std::string &s)
{
    size_t end=0;const auto n=std::stoull(s,&end);
    if(end!=s.size() || n>UINT32_MAX || s.empty() || s[0]=='-') throw std::runtime_error("invalid step count");
    return static_cast<natural_t>(n);
}
static int run(int argc,char **argv)
{
    bool resume=false,fields=false;
    std::string runId="000";
    natural_t endStep=NSTEPS,checkpointEvery=STAMP;
    for(int i=1;i<argc;++i)
    {
        const std::string arg=argv[i];
        if(arg=="--continue" || arg=="continue") resume=true;
        else if(arg=="--fields") fields=true;
        else if(arg=="--runId" || arg=="--steps" || arg=="--checkpoint-every")
        {
            if(++i>=argc) throw std::runtime_error("missing value for "+arg);
            if(arg=="--runId") runId=argv[i];
            else if(arg=="--steps") endStep=parseStep(argv[i]);
            else checkpointEvery=parseStep(argv[i]);
        }
        else if(arg.rfind("--runId=",0)==0) runId=arg.substr(8);
        else if(!arg.empty() && arg[0]!='-' && runId=="000") runId=arg;
        else throw std::runtime_error("unknown argument: "+arg);
    }
    if(!validRunId(runId) || checkpointEvery==0) throw std::runtime_error("invalid run ID or checkpoint interval");
    setSimulationRunId(runId);createOutputDirectories();
#if defined(CASE_RAYLEIGH_BENARD) || defined(CASE_JET)
    if(!resume && std::filesystem::exists(checkpointDirectory()) && !std::filesystem::is_empty(checkpointDirectory()))
        throw std::runtime_error("run already has checkpoints; choose another run ID or use --continue");
#endif
    real_t *moments=nullptr,*buffer=nullptr;
    constexpr size_t bytes=size_t(NUM_MOMENTS)*CELLS*sizeof(real_t);
    CUDA_CHECK(cudaMalloc(&moments,bytes));CUDA_CHECK(cudaMalloc(&buffer,bytes));
    constexpr dim3 grid(GRID_X,GRID_Y,GRID_Z),block(BLOCK_NX,BLOCK_NY,BLOCK_NZ);
    natural_t startStep=0;
#if defined(CASE_RAYLEIGH_BENARD)
    rbc::Statistics stats;
    if(resume) startStep=restoreCheckpoint(moments,stats);
#else
    CUDA_CHECK(initIRBCBoundaryTables());
    // Preserve the jet's full-field averages; RBC uses only reduced FP64 profiles.
    double *ta=nullptr,*ua=nullptr;real_t *tm=nullptr,*um=nullptr;uint64_t count=0;
    CUDA_CHECK(cudaMalloc(&ta,size_t(CELLS)*sizeof(double)));CUDA_CHECK(cudaMalloc(&ua,size_t(CELLS)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tm,size_t(CELLS)*sizeof(real_t)));CUDA_CHECK(cudaMalloc(&um,size_t(CELLS)*sizeof(real_t)));
    CUDA_CHECK(cudaMemset(ta,0,size_t(CELLS)*sizeof(double)));CUDA_CHECK(cudaMemset(ua,0,size_t(CELLS)*sizeof(double)));
    if(resume) startStep=restoreJetCheckpoint(moments,count,ta,ua);
#endif
    if(!resume)
    {
        initializeCase<<<grid,block>>>(moments,buffer);CUDA_CHECK(cudaGetLastError());
#if defined(CASE_RAYLEIGH_BENARD)
        applyNEEPlates<<<dim3((NX+31)/32,(NY+3)/4,2),dim3(32,4)>>>(moments);
#endif
    }
    if(endStep<startStep) throw std::runtime_error("requested end step precedes checkpoint");
    writeMetadata();
#if defined(CASE_RAYLEIGH_BENARD)
    stats.initializeSeries(resume,startStep);
    if(!resume) stats.sample(moments,0);
#endif
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout<<"case "<<ActiveCase::NAME<<" grid "<<NX<<'x'<<NY<<'x'<<NZ<<" RR"<<RR_ORDER
             <<" nu="<<VISCOSITY<<" kappa="<<THERMAL_DIFFUSIVITY<<" tau="<<TAU_SHEAR
             <<" tau_T="<<TAU_THERMAL<<" steps "<<startStep<<".."<<endStep<<'\n';
    const auto begin=std::chrono::steady_clock::now();
    for(natural_t step=startStep+1;step<=endStep;++step)
    {
        streamCollide<<<grid,block>>>(moments,buffer,step-1);
#if defined(CASE_RAYLEIGH_BENARD)
        applyNEEPlates<<<dim3((NX+31)/32,(NY+3)/4,2),dim3(32,4)>>>(buffer);
#endif
        std::swap(moments,buffer);CUDA_CHECK(cudaGetLastError());
#if defined(CASE_RAYLEIGH_BENARD)
        // Fixed lattice-step sampling; first average is strictly after the transient.
        if((step<=ActiveCase::timeAverageStart && step%ActiveCase::timeAverageStride==0) ||
           (step>ActiveCase::timeAverageStart && (step-ActiveCase::timeAverageStart)%ActiveCase::timeAverageStride==0))
            stats.sample(moments,step);
#else
        if(ActiveCase::timeAverageEnabled && step>ActiveCase::timeAverageStart &&
           (step-ActiveCase::timeAverageStart)%ActiveCase::timeAverageStride==0)
        {accumulateThermalTimeAverage<<<(CELLS+255)/256,256>>>(moments,ta,ua);++count;}
#endif
        if(step%checkpointEvery==0 || step==endStep)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
#if defined(CASE_RAYLEIGH_BENARD)
            saveCheckpoint(moments,step,stats);stats.exportAverages();writeMetadata(stats.count);
#else
            saveJetCheckpoint(moments,step,count,ta,ua);
#endif
            if(fields) writeOutput(moments,step);
            std::cout<<"step "<<step<<" / "<<endStep<<'\n';
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<"elapsed "<<elapsed<<" s; "<<double(CELLS)*(endStep-startStep)/elapsed/1.e6<<" MLUPS\n";
#if defined(CASE_RAYLEIGH_BENARD)
    stats.exportAverages();writeMetadata(stats.count);
    std::cout<<"averaged samples "<<stats.count<<'\n';
#else
    if(count)
    {
        finalizeThermalTimeAverage<<<(CELLS+255)/256,256>>>(ta,ua,tm,um,count);
        CUDA_CHECK(cudaGetLastError());writeScalarOutput(tm,"temperatureMean");writeScalarOutput(um,"uzMean");
    }
    writeMetadata(count);cudaFree(ta);cudaFree(ua);cudaFree(tm);cudaFree(um);
#endif
    CUDA_CHECK(cudaFree(moments));CUDA_CHECK(cudaFree(buffer));return 0;
}
int main(int argc,char **argv)
{
    try {return run(argc,argv);}
    catch(const std::exception &e) {std::cerr<<"thermal-mrlbm: "<<e.what()<<'\n';return EXIT_FAILURE;}
}
