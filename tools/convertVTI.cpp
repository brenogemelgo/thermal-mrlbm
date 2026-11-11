// Standalone CPU converter. It shares neither CUDA kernels nor solver state.
#include <hdf5.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <signal.h>
#include <poll.h>
#include <unistd.h>

namespace fs = std::filesystem;
static void require(bool ok, const std::string &message)
{ if (!ok) throw std::runtime_error(message); }
static void hcheck(herr_t value)
{ require(value >= 0, "HDF5 read failed"); }
struct Handle
{
    hid_t id; herr_t (*close)(hid_t);
    Handle(hid_t value, herr_t (*closer)(hid_t)):id(value),close(closer)
    { require(id >= 0, "cannot open HDF5 object"); }
    ~Handle() { close(id); }
    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;
    operator hid_t() const { return id; }
};
static std::string attribute(hid_t object, const char *name)
{
    require(H5Aexists(object, name) > 0, std::string("missing attribute: ")+name);
    Handle attr(H5Aopen(object, name, H5P_DEFAULT), H5Aclose);
    Handle type(H5Aget_type(attr), H5Tclose);
    Handle space(H5Aget_space(attr), H5Sclose);
    require(H5Sget_simple_extent_npoints(space)==1 && H5Tget_class(type)==H5T_STRING,
            std::string("expected scalar string attribute: ")+name);
    if (H5Tis_variable_str(type) > 0)
    {
        char *raw=nullptr; hcheck(H5Aread(attr,type,&raw));
        const std::string result=raw ? raw : "";
        hcheck(H5free_memory(raw)); return result;
    }
    const size_t bytes=H5Tget_size(type);
    require(bytes>0 && bytes<1024*1024,"invalid HDF5 attribute length");
    std::vector<char> value(bytes+1,0); hcheck(H5Aread(attr,type,value.data()));
    return std::string(value.data());
}
static std::vector<hsize_t> shape(hid_t dataset)
{
    Handle space(H5Dget_space(dataset),H5Sclose);
    const int rank=H5Sget_simple_extent_ndims(space);
    require(rank>=0,"invalid dataset rank");
    std::vector<hsize_t> dims(rank);
    hcheck(H5Sget_simple_extent_dims(space,dims.data(),nullptr)); return dims;
}
static uint64_t multiply(uint64_t a,uint64_t b)
{
    require(!b || a<=std::numeric_limits<uint64_t>::max()/b,"volume size overflow");
    return a*b;
}
static uint64_t stepNumber(const std::string &text)
{
    require(!text.empty() && text.find_first_not_of("0123456789")==std::string::npos,"invalid timestep");
    size_t end=0; const auto number=std::stoull(text,&end);
    require(end==text.size(),"invalid timestep"); return number;
}
static void plane(hid_t dataset,hid_t type,hsize_t z,hsize_t ny,hsize_t nx,void *buffer)
{
    Handle fileSpace(H5Dget_space(dataset),H5Sclose);
    const hsize_t offset[]={z,0,0}, count[]={1,ny,nx};
    hcheck(H5Sselect_hyperslab(fileSpace,H5S_SELECT_SET,offset,nullptr,count,nullptr));
    Handle memory(H5Screate_simple(3,count,nullptr),H5Sclose);
    hcheck(H5Dread(dataset,type,memory,fileSpace,H5P_DEFAULT,buffer));
}
static void writeBytes(std::ostream &out,const void *data,size_t bytes)
{
    require(bytes<=size_t(std::numeric_limits<std::streamsize>::max()),"write size overflow");
    out.write(static_cast<const char *>(data),static_cast<std::streamsize>(bytes));
    require(bool(out),"VTI write failed (check available storage)");
}
static void writeLength(std::ostream &out,uint64_t length)
{
    unsigned char bytes[8];
    for(int i=0;i<8;++i) bytes[i]=static_cast<unsigned char>((length>>(8*i))&255);
    writeBytes(out,bytes,8);
}
struct Options
{
    fs::path input,results="results";
    std::set<uint64_t> steps;
    bool skipExisting=false;
    unsigned jobs=static_cast<unsigned>(std::min<long>(4,std::max<long>(1,sysconf(_SC_NPROCESSORS_ONLN))));
};
struct OutputLock
{
    int descriptor=-1;
    explicit OutputLock(const fs::path &destination)
    {
        fs::create_directories(destination.parent_path());
        const auto path=destination.string()+".lock";
        descriptor=::open(path.c_str(),O_CREAT|O_RDWR,0666);
        if(descriptor<0) throw std::runtime_error("cannot open VTI output lock");
        if(flock(descriptor,LOCK_EX|LOCK_NB)!=0)
        { ::close(descriptor);descriptor=-1;throw std::runtime_error("conflicting conversion for "+destination.string()); }
    }
    ~OutputLock() { if(descriptor>=0) ::close(descriptor); }
};
static bool validRunId(const std::string &id)
{
    return !id.empty() && id!="." && id!=".." && id.find('/')==id.npos && id.find('\\')==id.npos;
}
// Return false for non-volume HDF5 files. Dataset content is never loaded here.
static bool convert(const fs::path &source,const Options &options,std::set<fs::path> &destinations,
                    std::set<uint64_t> &foundSteps)
{
    Handle file(H5Fopen(source.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
    if(H5Lexists(file,"fields",H5P_DEFAULT)<=0) return false;
    Handle fields(H5Gopen2(file,"fields",H5P_DEFAULT),H5Gclose);
    if(H5Lexists(fields,"temperature",H5P_DEFAULT)<=0) return false;
    Handle temperature(H5Dopen2(fields,"temperature",H5P_DEFAULT),H5Dclose);
    const auto dims=shape(temperature);
    if(dims.size()==2) return false; // Slices are not 3D volume renderings.
    require(dims.size()==3 && dims[0] && dims[1] && dims[2],"expected a nonempty ZYX temperature volume");
    const auto step=stepNumber(attribute(file,"step"));
    if(!options.steps.empty() && !options.steps.count(step)) return false;
    require(attribute(file,"axis_order")=="zyx","expected ZYX axis order");
    const bool hasVelocity=H5Lexists(fields,"ux",H5P_DEFAULT)>0;
    require((H5Lexists(fields,"uy",H5P_DEFAULT)>0)==hasVelocity &&
            (H5Lexists(fields,"uz",H5P_DEFAULT)>0)==hasVelocity,"incomplete velocity vector");
    if(hasVelocity)
        require(attribute(file,"velocity_convention")=="physical_u","expected physical velocities; refusing an ambiguous force shift");
    const std::string runId=attribute(file,"run_id");
    require(validRunId(runId),"invalid original run ID");
    Handle ttype(H5Dget_type(temperature),H5Tclose);
    const size_t bytes=H5Tget_size(ttype);
    require(H5Tget_class(ttype)==H5T_FLOAT && (bytes==4 || bytes==8),"expected FP32/FP64 fields");
    std::vector<std::unique_ptr<Handle>> velocity;
    if(hasVelocity)
        for(const auto name:{"ux","uy","uz"})
            velocity.emplace_back(new Handle(H5Dopen2(fields,name,H5P_DEFAULT),H5Dclose));
    for(const auto &data:velocity)
    {
        Handle type(H5Dget_type(*data),H5Tclose);
        require(shape(*data)==dims && H5Tget_class(type)==H5T_FLOAT && H5Tget_size(type)==bytes,
                "inconsistent velocity shape/precision");
    }
    const uint64_t planeCells=multiply(dims[1],dims[2]);
    const uint64_t planeBytes=multiply(planeCells,bytes);
    const uint64_t scalarBytes=multiply(planeBytes,dims[0]);
    require(scalarBytes<=(std::numeric_limits<uint64_t>::max()-16)/4,"VTI payload size overflow");
    require(planeBytes<=std::numeric_limits<size_t>::max()/3,"plane allocation overflow");
    const hid_t diskType=bytes==4 ? H5T_IEEE_F32LE : H5T_IEEE_F64LE;
    const char *vtkType=bytes==4 ? "Float32" : "Float64";
    std::ostringstream name; name<<"step_";
    const auto number=std::to_string(step); name<<std::string(number.size()<9?9-number.size():0,'0')<<number<<".vti";
    const auto destination=options.results/runId/"vtis"/name.str();
    require(destinations.insert(fs::absolute(destination).lexically_normal()).second,"duplicate run ID/timestep in input tree");
    const std::string extent="0 "+std::to_string(dims[2]-1)+" 0 "+std::to_string(dims[1]-1)+" 0 "+std::to_string(dims[0]-1);
    std::ostringstream header;
    header<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
          <<"  <ImageData WholeExtent=\""<<extent<<"\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
          <<"    <Piece Extent=\""<<extent<<"\">\n      <PointData Scalars=\"temperature\""
          <<(hasVelocity ? " Vectors=\"velocity\"" : "")<<">\n"
          <<"        <DataArray type=\""<<vtkType<<"\" Name=\"temperature\" NumberOfComponents=\"1\" format=\"appended\" offset=\"0\"/>\n";
    if(hasVelocity)
        header<<"        <DataArray type=\""<<vtkType<<"\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\""<<scalarBytes+8<<"\"/>\n";
    header<<"      </PointData>\n      <CellData/>\n    </Piece>\n  </ImageData>\n  <AppendedData encoding=\"raw\">\n_";
    const std::string prefix=header.str(),suffix="\n  </AppendedData>\n</VTKFile>\n";
    foundSteps.insert(step);
    OutputLock outputLock(destination);
    if(fs::exists(destination))
    {
        require(options.skipExisting,"refusing to overwrite "+destination.string()+" (use --skip-existing to resume)");
        require(fs::file_size(destination)==prefix.size()+(hasVelocity ? 16+4*scalarBytes : 8+scalarBytes)+suffix.size(),"existing VTI has an incompatible size");
        std::cout<<"preserved "<<destination<<'\n'; return true;
    }
    fs::create_directories(destination.parent_path());
    auto temporary=destination; temporary+=".tmp";
    try
    {
        std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
        require(bool(out),"cannot create "+temporary.string());
        writeBytes(out,prefix.data(),prefix.size()); writeLength(out,scalarBytes);
        std::vector<unsigned char> scalar(static_cast<size_t>(planeBytes));
        for(hsize_t z=0;z<dims[0];++z)
        { plane(temperature,diskType,z,dims[1],dims[2],scalar.data()); writeBytes(out,scalar.data(),scalar.size()); }
        if(hasVelocity)
        {
            writeLength(out,3*scalarBytes);
            std::array<std::vector<unsigned char>,3> component;
            for(auto &values:component) values.resize(static_cast<size_t>(planeBytes));
            std::vector<unsigned char> packed(static_cast<size_t>(3*planeBytes));
            for(hsize_t z=0;z<dims[0];++z)
            {
                for(int d=0;d<3;++d) plane(*velocity[d],diskType,z,dims[1],dims[2],component[d].data());
                for(size_t i=0;i<planeCells;++i) for(size_t d=0;d<3;++d)
                    std::memcpy(packed.data()+(3*i+d)*bytes,component[d].data()+i*bytes,bytes);
                writeBytes(out,packed.data(),packed.size());
            }
        }
        writeBytes(out,suffix.data(),suffix.size()); out.close(); require(bool(out),"VTI close failed");
        fs::rename(temporary,destination);
    }
    catch(...) { fs::remove(temporary); throw; }
    std::cout<<destination<<'\n'; return true;
}
struct Lock
{
    int descriptor=-1;
    explicit Lock(const fs::path &root)
    {
        fs::create_directories(root);
        descriptor=::open((root/".convertVTI.lock").c_str(),O_CREAT|O_RDWR,0666);
        if(descriptor<0) throw std::runtime_error("cannot open converter lock");
        if(flock(descriptor,LOCK_EX|LOCK_NB)!=0)
        { ::close(descriptor);descriptor=-1;throw std::runtime_error("another converter owns this results root"); }
    }
    ~Lock() { if(descriptor>=0) ::close(descriptor); }
};
static void pipeWrite(int fd,const void *data,size_t size)
{
    auto p=static_cast<const unsigned char *>(data);
    while(size)
    {
        const auto n=::write(fd,p,size);
        if(n<0 && errno==EINTR) continue;
        require(n>0,"conversion worker pipe write failed");p+=n;size-=n;
    }
}
static void pipeRead(int fd,void *data,size_t size)
{
    auto p=static_cast<unsigned char *>(data);
    while(size)
    {
        const auto n=::read(fd,p,size);
        if(n<0 && errno==EINTR) continue;
        require(n>0,"conversion worker failed; completed VTI files were preserved");p+=n;size-=n;
    }
}
static size_t convertFiles(const std::vector<fs::path> &files,const Options &options,
                           size_t first,size_t stride,std::set<uint64_t> &found)
{
    size_t count=0;std::set<fs::path> destinations;
    for(size_t i=first;i<files.size();i+=stride)
    {
        try { if(convert(files[i],options,destinations,found)) ++count; }
        catch(const std::exception &e) { throw std::runtime_error(files[i].string()+": "+e.what()); }
    }
    return count;
}
// Separate processes avoid serializing HDF5 decompression on libraries built
// without thread safety. No HDF5 objects are opened in the parent before fork.
static size_t parallelConvert(const std::vector<fs::path> &files,const Options &options,
                               std::set<uint64_t> &found)
{
    const size_t jobs=std::min<size_t>(options.jobs,files.size());
    std::cout<<"convertVTI: "<<jobs<<" CPU worker process(es)\n"<<std::flush;
    if(jobs==1) return convertFiles(files,options,0,1,found);
    struct Worker {pid_t pid;int pipe;};std::vector<Worker> workers;
    size_t converted=0;
    try
    {
        for(size_t i=0;i<jobs;++i)
        {
            int channel[2];require(::pipe(channel)==0,"cannot create conversion worker pipe");
            const pid_t pid=::fork();
            if(pid<0) {::close(channel[0]);::close(channel[1]);throw std::runtime_error("cannot fork conversion worker");}
            if(pid==0)
            {
                ::close(channel[0]);for(const auto &worker:workers) ::close(worker.pipe);
                try
                {
                    std::set<uint64_t> steps;
                    const uint64_t count=convertFiles(files,options,i,jobs,steps),n=steps.size();
                    pipeWrite(channel[1],&count,sizeof(count));pipeWrite(channel[1],&n,sizeof(n));
                    for(const auto step:steps) pipeWrite(channel[1],&step,sizeof(step));
                    std::cout.flush();std::cerr.flush();::_exit(0);
                }
                catch(const std::exception &e) {std::cerr<<"convertVTI worker: "<<e.what()<<'\n'<<std::flush;::_exit(1);}
            }
            ::close(channel[1]);workers.push_back({pid,channel[0]});
        }
        size_t remaining=workers.size();
        while(remaining)
        {
            std::vector<pollfd> ready;
            for(const auto &worker:workers) ready.push_back({worker.pipe,POLLIN|POLLHUP,0});
            int result;
            do {result=::poll(ready.data(),ready.size(),-1);} while(result<0 && errno==EINTR);
            require(result>0,"cannot monitor conversion workers");
            for(size_t j=0;j<workers.size();++j)
            {
                if(!ready[j].revents) continue;
                auto &worker=workers[j];
                uint64_t count=0,n=0;pipeRead(worker.pipe,&count,sizeof(count));pipeRead(worker.pipe,&n,sizeof(n));
                for(uint64_t i=0;i<n;++i) {uint64_t step;pipeRead(worker.pipe,&step,sizeof(step));found.insert(step);}
                ::close(worker.pipe);worker.pipe=-1;
                int status=0;pid_t waited;
                do {waited=::waitpid(worker.pid,&status,0);} while(waited<0 && errno==EINTR);
                worker.pid=-1;
                require(waited>0 && WIFEXITED(status) && WEXITSTATUS(status)==0,"conversion worker failed");
                converted+=count;--remaining;
            }
        }
    }
    catch(...)
    {
        for(auto &worker:workers) {if(worker.pid>0) ::kill(worker.pid,SIGTERM);if(worker.pipe>=0) ::close(worker.pipe);}
        for(auto &worker:workers) if(worker.pid>0) {while(::waitpid(worker.pid,nullptr,0)<0 && errno==EINTR) {}}
        throw;
    }
    return converted;
}
static int run(int argc,char **argv)
{
    Options options;
    for(int i=1;i<argc;++i)
    {
        const std::string arg=argv[i];
        if(arg=="--help" || arg=="-h")
        {
            std::cout<<"Usage: convertVTI <path/to/hdfs/location> [--jobs N] [--results-root DIR] [--step N] [--skip-existing]\n"
                     <<"Reads one file, a fields/run directory, or an HDF5 tree recursively.\n"
                     <<"Writes results/<original_run_id>/vtis/; CPU processes, default up to 4 workers.\n"
                     <<"No CUDA context or solver work; --jobs 1 selects serial conversion.\n";
            return 0;
        }
        if(arg=="--skip-existing") options.skipExisting=true;
        else if(arg=="--results-root" || arg=="--step" || arg=="--jobs")
        {
            require(++i<argc,"missing value for "+arg);
            if(arg=="--results-root") options.results=argv[i];
            else if(arg=="--step") options.steps.insert(stepNumber(argv[i]));
            else {const auto n=stepNumber(argv[i]);require(n>0 && n<=256,"--jobs must be between 1 and 256");options.jobs=static_cast<unsigned>(n);}
        }
        else if(!arg.empty() && arg[0]!='-' && options.input.empty()) options.input=arg;
        else throw std::runtime_error("unknown argument: "+arg);
    }
    require(!options.input.empty(),"usage: convertVTI <path/to/hdfs/location> [--help]");
    require(fs::exists(options.input),"HDF5 input location does not exist");
    if(fs::is_directory(options.input) && fs::is_directory(options.input/"hdf5"))
        options.input=fs::canonical(options.input/"hdf5");
    std::vector<fs::path> files;
    auto add=[&](const fs::path &path)
    { if(path.extension()==".h5" || path.extension()==".hdf5" || path.extension()==".hdf") files.push_back(path); };
    if(fs::is_regular_file(options.input)) add(options.input);
    else for(const auto &entry:fs::recursive_directory_iterator(options.input))
        if(entry.is_regular_file()) add(entry.path());
    std::sort(files.begin(),files.end());
    require(!files.empty(),"no HDF5 files found");
    Lock lock(options.results);std::set<uint64_t> foundSteps;
    const size_t converted=parallelConvert(files,options,foundSteps);
    require(converted>0,"no matching 3D temperature volumes found (statistics and slices are skipped)");
    require(options.steps.empty() || foundSteps==options.steps,"some requested timesteps were not found");
    std::cout<<converted<<" volume(s) converted/preserved\n"; return 0;
}
int main(int argc,char **argv)
{
    try { return run(argc,argv); }
    catch(const std::exception &e) { std::cerr<<"convertVTI: "<<e.what()<<'\n'; return 1; }
}
