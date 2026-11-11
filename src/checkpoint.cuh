#pragma once
#include "rbcStatistics.cuh"
#include <array>

// A checkpoint is one atomic transaction: solver moments, schedule, statistics,
// and floating point compensation terms. Legacy full-moment files are rejected.
struct CheckpointHeader
{
    char magic[8]={'T','M','R','L','B','M','2','\0'};
    uint32_t nx=NX,ny=NY,nz=NZ,fields=NUM_MOMENTS,precision=sizeof(real_t),rr=RR_ORDER;
    uint64_t step=0,start=ActiveCase::timeAverageStart,stride=ActiveCase::timeAverageStride;
    std::array<double,9> physics={double(VISCOSITY),double(THERMAL_DIFFUSIVITY),double(BETA_THERMAL),
        double(T_HOT),double(T_COLD),double(T_REF),double(THERMAL_GRAVITY_X),double(THERMAL_GRAVITY_Y),double(THERMAL_GRAVITY_Z)};
#if defined(RBC_VALIDATION_CASE)
    // Encode the manufactured physics in the existing header without changing
    // the production checkpoint ABI. Production and V1/V2/V3 cannot cross-load.
    CheckpointHeader()
    {
        fields=NUM_MOMENTS | (uint32_t(RBC_VALIDATION_CASE)<<24);
        // Unused buoyancy/gravity slots fingerprint manufactured source parameters.
        physics[2]=ActiveCase::MODE_AMPLITUDE;
        physics[6]=ActiveCase::PECLET;
        physics[7]=ActiveCase::CHANNEL_RE;
        physics[8]=ActiveCase::FORMULA_VERSION;
    }
#endif
};

template<class T> inline void checkpointWrite(std::ostream &out,const T *p,size_t n=1)
{
    out.write(reinterpret_cast<const char *>(p),n*sizeof(T));
    if(!out) throw std::runtime_error("checkpoint write failed");
}
template<class T> inline void checkpointRead(std::istream &in,T *p,size_t n=1)
{
    in.read(reinterpret_cast<char *>(p),n*sizeof(T));
    if(!in) throw std::runtime_error("truncated checkpoint");
}
static std::filesystem::path checkpointDirectory() {return getSimulationOutputDirectory()/"checkpoints";}
static std::filesystem::path checkpointHistoryDirectory()
{
    return canonicalOutputPath(getSimulationHdf5Directory()) ==
        canonicalOutputPath(getSimulationOutputDirectory()) ?
        checkpointDirectory() : getSimulationHdf5Directory()/"checkpoint_history";
}

static std::filesystem::path latestCheckpoint()
{
    std::filesystem::path latest;
    uint64_t latestStep=0;
    if(std::filesystem::exists(checkpointDirectory()))
        for(auto &e:std::filesystem::directory_iterator(checkpointDirectory()))
        {
            const auto stem=e.path().stem().string();
            if(e.path().extension()!=".chk" || stem.rfind("step_",0)!=0) continue;
            size_t end=0;
            const uint64_t step=std::stoull(stem.substr(5),&end);
            if(end!=stem.size()-5) continue;
            if(latest.empty() || step>latestStep) {latest=e.path();latestStep=step;}
        }
    if(latest.empty()) throw std::runtime_error("no version-2 checkpoint found (legacy moments cannot be resumed)");
    return latest;
}
static void checkpointMoments(std::ostream &out,const real_t *m)
{
    constexpr natural_t chunk=std::min(CELLS,32u*STRIDE);
    std::vector<real_t> field(chunk);
    for(natural_t f=0;f<NUM_MOMENTS;++f)
    {
        for(natural_t first=0;first<CELLS;first+=chunk)
        {
            const natural_t count=std::min(chunk,CELLS-first);
            moment_io::download(field.data(),m,f,first,count);
            checkpointWrite(out,field.data(),count);
        }
    }
}
static void restoreMoments(std::istream &in,real_t *m)
{
    constexpr natural_t chunk=std::min(CELLS,32u*STRIDE);
    std::vector<real_t> field(chunk);
    for(natural_t f=0;f<NUM_MOMENTS;++f)
    {
        for(natural_t first=0;first<CELLS;first+=chunk)
        {
            const natural_t count=std::min(chunk,CELLS-first);
            checkpointRead(in,field.data(),count);
            moment_io::upload(m,field.data(),f,first,count);
        }
    }
}
#if defined(CASE_RAYLEIGH_BENARD)
static void saveCheckpoint(const real_t *m,natural_t step,const rbc::Statistics &s)
{
    std::filesystem::create_directories(checkpointDirectory());
    auto path=checkpointDirectory()/(outputStepName(step)+".chk");auto tmp=path;tmp+=".tmp";
    std::ofstream out(tmp,std::ios::binary);CheckpointHeader h;h.step=step;checkpointWrite(out,&h);
    checkpointMoments(out,m);
    checkpointWrite(out,&s.count);checkpointWrite(out,&s.lastSample);checkpointWrite(out,&s.etaMin);
    checkpointWrite(out,s.mean.data(),s.mean.size());checkpointWrite(out,s.m2.data(),s.m2.size());
    checkpointWrite(out,s.sum.data(),s.sum.size());checkpointWrite(out,s.comp.data(),s.comp.size());
    out.close();if(!out) throw std::runtime_error("checkpoint close failed");
    // Commit a closed HDF5 history alongside the unchanged binary checkpoint.
    // The .chk rename is last: even a damaged in-flight statistics file can be
    // recovered without losing the history preceding a committed checkpoint.
    std::filesystem::create_directories(checkpointHistoryDirectory());
    const auto history=checkpointHistoryDirectory()/("statistics_"+outputStepName(step)+".h5");
    auto historyTmp=history;historyTmp+=".tmp";
    std::filesystem::copy_file(statisticsPath(),historyTmp,std::filesystem::copy_options::overwrite_existing);
    std::filesystem::rename(historyTmp,history);
    std::filesystem::rename(tmp,path);
}
static natural_t restoreCheckpoint(real_t *m,rbc::Statistics &s)
{
    const auto path=latestCheckpoint();std::ifstream in(path,std::ios::binary);
    CheckpointHeader h,expected;checkpointRead(in,&h);
    if(std::memcmp(h.magic,expected.magic,8) || h.nx!=NX || h.ny!=NY || h.nz!=NZ ||
       h.fields!=expected.fields || h.precision!=sizeof(real_t) || h.rr!=RR_ORDER || h.start!=expected.start ||
       h.stride!=expected.stride || h.physics!=expected.physics || h.step>UINT32_MAX)
        throw std::runtime_error("incompatible checkpoint moment format, grid, physics, RR order or averaging schedule");
    restoreMoments(in,m);
    checkpointRead(in,&s.count);checkpointRead(in,&s.lastSample);checkpointRead(in,&s.etaMin);
    checkpointRead(in,s.mean.data(),s.mean.size());checkpointRead(in,s.m2.data(),s.m2.size());
    checkpointRead(in,s.sum.data(),s.sum.size());checkpointRead(in,s.comp.data(),s.comp.size());
    if(in.peek()!=std::char_traits<char>::eof() || s.lastSample!=(s.count ? h.start+s.count*h.stride : 0) ||
       s.count!=(h.step>h.start ? (h.step-h.start)/h.stride : 0))
        throw std::runtime_error("invalid checkpoint statistics or file length");
    auto history=checkpointHistoryDirectory()/("statistics_"+outputStepName(h.step)+".h5");
    if(!std::filesystem::exists(history)) history=checkpointDirectory()/history.filename();
    if(std::filesystem::exists(history))
    {
        auto tmp=statisticsPath();tmp+=".restore";
        std::filesystem::copy_file(history,tmp,std::filesystem::copy_options::overwrite_existing);
        std::filesystem::rename(tmp,statisticsPath());
    }
    std::cout<<"restored "<<path<<" with "<<s.count<<" averaged samples\n";
    return static_cast<natural_t>(h.step);
}
#endif
#if defined(CASE_JET)
static void saveJetCheckpoint(const real_t *m,natural_t step,uint64_t count,const double *ta,const double *ua)
{
    std::filesystem::create_directories(checkpointDirectory());
    auto path=checkpointDirectory()/(outputStepName(step)+".chk"),tmp=path;tmp+=".tmp";
    std::ofstream out(tmp,std::ios::binary);CheckpointHeader h;h.step=step;checkpointWrite(out,&h);
    checkpointMoments(out,m);checkpointWrite(out,&count);
    std::vector<double> data(CELLS);
    for(auto p:{ta,ua})
    {
        outputCheckCuda(cudaMemcpy(data.data(),p,CELLS*sizeof(double),cudaMemcpyDeviceToHost),"checkpoint jet averages");
        checkpointWrite(out,data.data(),data.size());
    }
    out.close();if(!out) throw std::runtime_error("checkpoint close failed");std::filesystem::rename(tmp,path);
}
static natural_t restoreJetCheckpoint(real_t *m,uint64_t &count,double *ta,double *ua)
{
    std::ifstream in(latestCheckpoint(),std::ios::binary);CheckpointHeader h,e;checkpointRead(in,&h);
    if(std::memcmp(h.magic,e.magic,8) || h.nx!=e.nx || h.ny!=e.ny || h.nz!=e.nz || h.fields!=e.fields ||
       h.precision!=e.precision || h.rr!=e.rr || h.start!=e.start || h.stride!=e.stride || h.physics!=e.physics || h.step>UINT32_MAX)
        throw std::runtime_error("incompatible jet checkpoint");
    restoreMoments(in,m);checkpointRead(in,&count);std::vector<double> data(CELLS);
    for(auto p:{ta,ua})
    {
        checkpointRead(in,data.data(),data.size());
        outputCheckCuda(cudaMemcpy(p,data.data(),CELLS*sizeof(double),cudaMemcpyHostToDevice),"restore jet averages");
    }
    if(in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("invalid jet checkpoint length");
    return h.step;
}
#endif
