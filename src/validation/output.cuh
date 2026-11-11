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
    SliceOutput()
    {
        if(!moment_io::distributed())
            outputCheckCuda(cudaMalloc(&device,host.size()*sizeof(double)),"validation slice allocation");
    }
    ~SliceOutput() {cudaFree(device);}
    void save(const real_t *now,const real_t *old,natural_t step)
    {
        if(moment_io::distributed()) moment_io::distributed()->validation(host.data(),now,old);
        else
        {
            extractSlice<<<dim3((NX+127)/128,NZ),128>>>(now,old,device);
            outputCheckCuda(cudaGetLastError(),"extract validation slice");
            outputCheckCuda(cudaMemcpy(host.data(),device,host.size()*sizeof(double),cudaMemcpyDeviceToHost),"copy validation slice");
        }
        const auto dir=getSimulationHdf5Directory()/"validation";
        std::filesystem::create_directories(dir);
        const auto base=dir/("snapshot_"+std::to_string(step));
        auto path=base;path+=".h5";auto tmp=path;tmp+=".tmp";
        {
            h5out::Handle file(H5Fcreate(tmp.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT),H5Fclose);
            h5out::attribute(file,"schema","1");
            h5out::attribute(file,"case","V"+std::to_string(KIND));
            h5out::attribute(file,"step",std::to_string(step));
            h5out::attribute(file,"previous_step",std::to_string(step ? step-1 : 0));
            h5out::attribute(file,"nx",std::to_string(NX));h5out::attribute(file,"nz",std::to_string(NZ));
            h5out::attribute(file,"channels",std::to_string(SLICE_CHANNELS));
            h5out::Handle data(h5out::dataset(file,"moments",H5T_IEEE_F64LE,{SLICE_CHANNELS,NZ,NX},
                {1,std::min(64u,NZ),std::min(128u,NX)}),H5Dclose);
            h5out::check(H5Dwrite(data,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,host.data()));
            h5out::flush(file);
        }
        std::filesystem::rename(tmp,path);
    }
};
}
#endif
