#pragma once
#include "../output.cuh"
#include "../reconstruction.cuh"

#if defined(RBC_VALIDATION_CASE)
namespace validation
{
// Only an XZ slice of the two existing SoA buffers is staged. All manufactured
// fields are exactly invariant in Y; channel 28 verifies that invariant over Y.
constexpr size_t SLICE_CELLS=size_t(NX)*NZ;
constexpr size_t SLICE_CHANNELS=2*NUM_MOMENTS+1;
__global__ void extractSlice(const real_t *now,const real_t *old,double *slice)
{
    const natural_t x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y;
    if(x>=NX) return;
    const size_t i=x+size_t(z)*NX;
    double mismatch=0;
    for(natural_t f=0;f<NUM_MOMENTS;++f)
    {
        const double a=loadMoment(now,global3(x,0,z),f),b=loadMoment(old,global3(x,0,z),f);
        slice[size_t(f)*SLICE_CELLS+i]=a;
        slice[size_t(f+NUM_MOMENTS)*SLICE_CELLS+i]=b;
        for(natural_t y=1;y<NY;++y)
        {
            const double c=loadMoment(now,global3(x,y,z),f),d=loadMoment(old,global3(x,y,z),f);
            if(!isfinite(c) || !isfinite(d)) mismatch=INFINITY;
            else
            {
                mismatch=fmax(mismatch,fabs(c-a));
                mismatch=fmax(mismatch,fabs(d-b));
            }
        }
    }
    slice[2*NUM_MOMENTS*SLICE_CELLS+i]=mismatch;
}
struct SliceOutput
{
    double *device=nullptr;
    std::vector<double> host=std::vector<double>(SLICE_CHANNELS*SLICE_CELLS);
    SliceOutput() {outputCheckCuda(cudaMalloc(&device,host.size()*sizeof(double)),"validation slice allocation");}
    ~SliceOutput() {cudaFree(device);}
    void save(const real_t *now,const real_t *old,natural_t step)
    {
        extractSlice<<<dim3((NX+127)/128,NZ),128>>>(now,old,device);
        outputCheckCuda(cudaGetLastError(),"extract validation slice");
        outputCheckCuda(cudaMemcpy(host.data(),device,host.size()*sizeof(double),cudaMemcpyDeviceToHost),"copy validation slice");
        const auto dir=getSimulationOutputDirectory()/"validation";
        std::filesystem::create_directories(dir);
        const auto base=dir/("snapshot_"+std::to_string(step));
        auto bin=base;bin+=".bin";auto tmp=bin;tmp+=".tmp";
        std::ofstream out(tmp,std::ios::binary);
        out.write(reinterpret_cast<const char *>(host.data()),host.size()*sizeof(double));out.close();
        if(!out) throw std::runtime_error("validation slice write failed");
        std::filesystem::rename(tmp,bin);
        auto meta=base;meta+=".json";auto mt=meta;mt+=".tmp";
        std::ofstream j(mt);
        j<<"{\"schema\":1,\"case\":\"V"<<KIND<<"\",\"step\":"<<step
         <<",\"previous_step\":"<<(step ? step-1 : 0)<<",\"nx\":"<<NX<<",\"nz\":"<<NZ
         <<",\"channels\":"<<SLICE_CHANNELS<<",\"dtype\":\"native_float64\"}\n";
        j.close();if(!j) throw std::runtime_error("validation slice manifest write failed");
        std::filesystem::rename(mt,meta); // manifest is the commit record
    }
};
}
#endif
