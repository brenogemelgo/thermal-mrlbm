#pragma once
#include "output.cuh"
#include "reconstruction.cuh"
#include <array>
#include <limits>

#if defined(CASE_RAYLEIGH_BENARD)
namespace rbc
{
// Per-plane, spatial first/second raw moments. No persistent full-field statistics.
enum Stat {T,U,V,W,T2,U2,V2,W2,WT,COND,KE,EU,ET,ETA,ETA_MIN,UMAX,RHO_SUM,BAD,NS};
constexpr int THREADS=256;
constexpr natural_t TILES=(STRIDE+THREADS-1)/THREADS;
constexpr size_t PROFILE_SIZE=NS*NZ;

__device__ inline double primitive(const real_t *m,int x,int y,int z,int f)
{
    x=(x+NX)%NX; y=(y+NY)%NY;
    const natural_t i=global3(x,y,z);
    double value=loadMoment(m,i,f);
    if(f>=UX && f<=UZ)
    {
        real_t a[3]; acceleration(loadMoment(m,i,TEMP),z>0 && z<int(NZ)-1,a);
        value-=double(a[f-UX])/2;
    }
    return value;
}
__device__ inline double derivative(const real_t *m,int x,int y,int z,int f,int axis)
{
    if(axis==0) return (primitive(m,x+1,y,z,f)-primitive(m,x-1,y,z,f))/2;
    if(axis==1) return (primitive(m,x,y+1,z,f)-primitive(m,x,y-1,z,f))/2;
    if(z==0) return (-3*primitive(m,x,y,z,f)+4*primitive(m,x,y,1,f)-primitive(m,x,y,2,f))/2;
    if(z==NZ-1) return (3*primitive(m,x,y,z,f)-4*primitive(m,x,y,z-1,f)+primitive(m,x,y,z-2,f))/2;
    return (primitive(m,x,y,z+1,f)-primitive(m,x,y,z-1,f))/2;
}
// First reduction stage: one warp-contiguous XY tile, one Z plane per block.
__global__ void reduceTiles(const real_t *m,double *partial)
{
    __shared__ double shared[THREADS];
    const int lane=threadIdx.x, xy=blockIdx.x*THREADS+lane, z=blockIdx.y;
    double s[NS]={};
    s[ETA_MIN]=INFINITY;
    if(xy<STRIDE)
    {
        const int x=xy%NX,y=xy/NX;
        const double t=primitive(m,x,y,z,TEMP);
        const double u=primitive(m,x,y,z,UX),v=primitive(m,x,y,z,UY),w=primitive(m,x,y,z,UZ);
        s[T]=t;s[U]=u;s[V]=v;s[W]=w;
        s[T2]=t*t;s[U2]=u*u;s[V2]=v*v;s[W2]=w*w;
        s[WT]=w*(t-T_REF);s[KE]=(u*u+v*v+w*w)/2;
        double grad[3][3];
        for(int a=0;a<3;++a) for(int b=0;b<3;++b)
            grad[a][b]=derivative(m,x,y,z,UX+a,b);
        for(int a=0;a<3;++a)
        {
            const double g=derivative(m,x,y,z,TEMP,a);
            s[ET]+=double(THERMAL_DIFFUSIVITY)*g*g;
            for(int b=0;b<3;++b)
            {
                const double strain=grad[a][b]+grad[b][a];
                s[EU]+=double(VISCOSITY)*strain*strain/2;
            }
        }
        s[COND]=-double(THERMAL_DIFFUSIVITY)*derivative(m,x,y,z,TEMP,2);
        s[ETA]=s[EU]>0 ? pow(double(VISCOSITY)*VISCOSITY*VISCOSITY/s[EU],0.25) : INFINITY;
        s[ETA_MIN]=s[ETA];s[UMAX]=sqrt(u*u+v*v+w*w);
        s[RHO_SUM]=loadMoment(m,global3(x,y,z),RHO);
        s[BAD]=!isfinite(t+u+v+w+s[EU]+s[ET]+s[RHO_SUM]) || s[RHO_SUM]<=0;
    }
    // One small shared array limits occupancy pressure; sum each observable in FP64.
    for(int f=0;f<NS;++f)
    {
        shared[lane]=s[f];__syncthreads();
        for(int stride=THREADS/2;stride;stride/=2)
        {
            if(lane<stride)
                shared[lane]=f==ETA_MIN ? fmin(shared[lane],shared[lane+stride]) :
                             f==UMAX ? fmax(shared[lane],shared[lane+stride]) :
                             shared[lane]+shared[lane+stride];
            __syncthreads();
        }
        if(lane==0) partial[(size_t(z)*NS+f)*TILES+blockIdx.x]=shared[0];
        __syncthreads();
    }
}
__global__ void reducePlanes(const double *partial,double *profiles)
{
    const int f=blockIdx.x%NS,z=blockIdx.x/NS;
    __shared__ double shared[THREADS];
    double v=f==ETA_MIN ? INFINITY : 0;
    for(int k=threadIdx.x;k<TILES;k+=THREADS)
    {
        const double a=partial[(size_t(z)*NS+f)*TILES+k];
        v=f==ETA_MIN ? fmin(v,a) : f==UMAX ? fmax(v,a) : v+a;
    }
    shared[threadIdx.x]=v;__syncthreads();
    for(int stride=THREADS/2;stride;stride/=2)
    {
        if(threadIdx.x<stride)
            shared[threadIdx.x]=f==ETA_MIN ? fmin(shared[threadIdx.x],shared[threadIdx.x+stride]) :
                f==UMAX ? fmax(shared[threadIdx.x],shared[threadIdx.x+stride]) :
                shared[threadIdx.x]+shared[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0) profiles[z*NS+f]=(f==ETA_MIN || f==UMAX) ? shared[0] : shared[0]/STRIDE;
}

struct Statistics
{
    double *partial=nullptr,*deviceProfile=nullptr;
    std::vector<double> profile=std::vector<double>(PROFILE_SIZE);
    std::vector<double> sum=std::vector<double>(PROFILE_SIZE),comp=std::vector<double>(PROFILE_SIZE);
    std::array<double,5> mean{},m2{};
    uint64_t count=0,lastSample=0;
    double etaMin=INFINITY;
    Statistics()
    {
        outputCheckCuda(cudaMalloc(&partial,size_t(TILES)*PROFILE_SIZE*sizeof(double)),"allocate reduction tiles");
        outputCheckCuda(cudaMalloc(&deviceProfile,PROFILE_SIZE*sizeof(double)),"allocate profiles");
    }
    ~Statistics() {cudaFree(partial);cudaFree(deviceProfile);}
    static std::filesystem::path seriesPath() {return getSimulationOutputDirectory()/"rbc_timeseries.csv";}
    static const char *seriesHeader()
    {
        return "step,t_ff,averaging,nu_bottom,nu_top,nu_volume,nu_kinetic,nu_thermal,temperature,urms,vrms,wrms,trms,kinetic_energy,epsilon_u,epsilon_t,nu_z_range,delta_t_cells,eta_min,max_velocity,mass_density\n";
    }
    void initializeSeries(bool resume,natural_t step)
    {
        // Discard diagnostics newer than the committed checkpoint; never duplicate samples.
        std::vector<std::string> keep;
        if(resume)
        {
            std::ifstream in(seriesPath());std::string line;std::getline(in,line);
            while(std::getline(in,line))
                if(std::stoull(line.substr(0,line.find(',')))<=step) keep.push_back(line);
        }
        std::ofstream out(seriesPath());out<<seriesHeader();for(auto &s:keep) out<<s<<'\n';
        if(!out) throw std::runtime_error("cannot initialize RBC time series");
    }
    double volume(int f,const std::vector<double> &p) const
    {
        double v=0;
        for(natural_t z=0;z<NZ;++z) v+=(z==0 || z==NZ-1 ? 0.5 : 1)*p[z*NS+f];
        return v/ActiveCase::H;
    }
    std::array<double,5> nusselt(const std::vector<double> &p) const
    {
        const double h=ActiveCase::H,dt=ActiveCase::DELTA_T,k=THERMAL_DIFFUSIVITY,nu=VISCOSITY;
        return {p[COND]*h/(k*dt),p[(NZ-1)*NS+COND]*h/(k*dt),
            1+h*volume(WT,p)/(k*dt),
            ActiveCase::RA>0 ? 1+volume(EU,p)*h*h*h*h*ActiveCase::PR*ActiveCase::PR/(nu*nu*nu*ActiveCase::RA) : 1,
            volume(ET,p)*h*h/(k*dt*dt)};
    }
    void sample(const real_t *m,natural_t step)
    {
        reduceTiles<<<dim3(TILES,NZ),THREADS>>>(m,partial);
        reducePlanes<<<PROFILE_SIZE,THREADS>>>(partial,deviceProfile);
        outputCheckCuda(cudaGetLastError(),"RBC reductions");
        outputCheckCuda(cudaMemcpy(profile.data(),deviceProfile,PROFILE_SIZE*sizeof(double),cudaMemcpyDeviceToHost),"copy profiles");
        if(volume(BAD,profile)>0) throw std::runtime_error("nonfinite RBC state or nonpositive density");
        const auto nu=nusselt(profile);
        const bool avg=step>ActiveCase::timeAverageStart;
        if(avg)
        {
            ++count;lastSample=step;
            for(size_t i=0;i<sum.size();++i)
            {
                // Kahan accumulation for long averages, except extrema and eta=inf.
                if(i%NS==ETA_MIN) {sum[i]=count==1 ? profile[i] : std::min(sum[i],profile[i]);continue;}
                if(i%NS==UMAX) {sum[i]=std::max(sum[i],profile[i]);continue;}
                if(!std::isfinite(profile[i])) {sum[i]=profile[i];comp[i]=0;continue;}
                if(!std::isfinite(sum[i])) continue;
                const double y=profile[i]-comp[i],tmp=sum[i]+y;
                comp[i]=(tmp-sum[i])-y;sum[i]=tmp;
            }
            for(int i=0;i<5;++i) {const double d=nu[i]-mean[i];mean[i]+=d/count;m2[i]+=d*(nu[i]-mean[i]);}
        }
        double nzMin=INFINITY,nzMax=-INFINITY,em=INFINITY,um=0;
        for(natural_t z=0;z<NZ;++z)
        {
            const double nz=(profile[z*NS+WT]+profile[z*NS+COND])*ActiveCase::H/THERMAL_DIFFUSIVITY;
            nzMin=std::min(nzMin,nz);nzMax=std::max(nzMax,nz);
            em=std::min(em,profile[z*NS+ETA_MIN]);um=std::max(um,profile[z*NS+UMAX]);
        }
        if(avg) etaMin=std::min(etaMin,em);
        const double tm=volume(T,profile),u=volume(U,profile),v=volume(V,profile),w=volume(W,profile);
        auto rms=[](double a,double b){return std::sqrt(std::max(0.0,a-b*b));};
        std::ofstream out(seriesPath(),std::ios::app);out<<std::setprecision(17)<<step<<','<<step/ActiveCase::T_FREEFALL<<','<<avg;
        for(double a:nu) out<<','<<a;
        out<<','<<tm<<','<<rms(volume(U2,profile),u)<<','<<rms(volume(V2,profile),v)<<','<<rms(volume(W2,profile),w)
           <<','<<rms(volume(T2,profile),tm)<<','<<volume(KE,profile)<<','<<volume(EU,profile)<<','<<volume(ET,profile)
           <<','<<nzMax-nzMin<<','<<ActiveCase::H/(nu[0]+nu[1])<<','<<em<<','<<um<<','<<volume(RHO_SUM,profile)<<'\n';
        if(!out) throw std::runtime_error("cannot append RBC diagnostics");
    }
    void exportAverages() const
    {
        std::ofstream out(getSimulationOutputDirectory()/"rbc_profiles.csv");
        out<<"z,z_over_h,temperature,ux,uy,uz,trms,urms,vrms,wrms,convective_flux,conductive_flux,nu_z,kinetic_energy,epsilon_u,epsilon_t,eta_mean,eta_min,delta_over_eta_min\n"<<std::setprecision(17);
        if(count==0) return;
        std::vector<double> p=sum;
        for(size_t i=0;i<p.size();++i) if(i%NS!=ETA_MIN && i%NS!=UMAX) p[i]/=count;
        for(natural_t z=0;z<NZ;++z)
        {
            const double *a=&p[z*NS];out<<z<<','<<z/double(ActiveCase::H);
            for(int f:{T,U,V,W}) out<<','<<a[f];
            for(auto pair:{std::pair<int,int>{T2,T},{U2,U},{V2,V},{W2,W}})
                out<<','<<std::sqrt(std::max(0.0,a[pair.first]-a[pair.second]*a[pair.second]));
            out<<','<<a[WT]<<','<<a[COND]<<','<<(a[WT]+a[COND])*ActiveCase::H/THERMAL_DIFFUSIVITY;
            for(int f:{KE,EU,ET,ETA,ETA_MIN}) out<<','<<a[f];
            out<<','<<1/a[ETA_MIN]<<'\n';
        }
        std::ofstream c(getSimulationOutputDirectory()/"rbc_convergence.csv");
        c<<"estimator,samples,mean,sample_stddev,naive_stderr\n"<<std::setprecision(17);
        const char *names[]={"bottom","top","volume","kinetic","thermal"};
        for(int i=0;i<5;++i)
        {
            const double sd=count>1 ? std::sqrt(m2[i]/(count-1)) : 0;
            c<<names[i]<<','<<count<<','<<mean[i]<<','<<sd<<','<<sd/std::sqrt(double(count))<<'\n';
        }
        // Correlation-aware block errors are computed by postRayleigh.py.
    }
};
}
#endif
