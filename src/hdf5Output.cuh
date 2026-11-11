#pragma once
#include <hdf5.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdint>

// Host-only HDF5 helpers. No populations or persistent solver fields are added.
namespace h5out
{
inline int &compression() {static int value=1;return value;}
inline double &ioSeconds() {static double value=0;return value;}
struct OutputTimer
{
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    ~OutputTimer() {ioSeconds()+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
};
inline void check(herr_t status) {if(status<0) throw std::runtime_error("HDF5 operation failed");}
struct Handle
{
    hid_t id; herr_t (*close)(hid_t);
    Handle(hid_t value,herr_t (*closer)(hid_t)):id(value),close(closer)
    {if(id<0) throw std::runtime_error("HDF5 open/create failed");}
    ~Handle() {if(id>=0) close(id);}
    Handle(const Handle&)=delete;Handle& operator=(const Handle&)=delete;
    operator hid_t() const {return id;}
};
inline void flush(hid_t file) {check(H5Fflush(file,H5F_SCOPE_GLOBAL));}
inline bool exists(hid_t parent,const std::string &name) {return H5Lexists(parent,name.c_str(),H5P_DEFAULT)>0;}
inline void attribute(hid_t parent,const std::string &name,const std::string &value)
{
    if(H5Aexists(parent,name.c_str())>0) check(H5Adelete(parent,name.c_str()));
    Handle type(H5Tcopy(H5T_C_S1),H5Tclose);check(H5Tset_size(type,value.size()+1));
    Handle space(H5Screate(H5S_SCALAR),H5Sclose);
    Handle attr(H5Acreate2(parent,name.c_str(),type,space,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
    check(H5Awrite(attr,type,value.c_str()));
}
inline hid_t group(hid_t parent,const char *name)
{
    return exists(parent,name) ? H5Gopen2(parent,name,H5P_DEFAULT) :
        H5Gcreate2(parent,name,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
}
inline hid_t file(const std::filesystem::path &path)
{
    return std::filesystem::exists(path) ? H5Fopen(path.c_str(),H5F_ACC_RDWR,H5P_DEFAULT) :
        H5Fcreate(path.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT);
}
inline hid_t dataset(hid_t parent,const char *name,hid_t type,const std::vector<hsize_t> &dims,
                     const std::vector<hsize_t> &chunks,const std::vector<hsize_t> &maxdims={})
{
    Handle space(H5Screate_simple(int(dims.size()),dims.data(),maxdims.empty()?nullptr:maxdims.data()),H5Sclose);
    Handle prop(H5Pcreate(H5P_DATASET_CREATE),H5Pclose);
    check(H5Pset_chunk(prop,int(chunks.size()),chunks.data()));
    if(compression()>0) {check(H5Pset_shuffle(prop));check(H5Pset_deflate(prop,compression()));}
    check(H5Pset_fletcher32(prop));
    return H5Dcreate2(parent,name,type,space,H5P_DEFAULT,prop,H5P_DEFAULT);
}
inline std::vector<hsize_t> dimensions(hid_t data)
{
    Handle space(H5Dget_space(data),H5Sclose);
    std::vector<hsize_t> dims(H5Sget_simple_extent_ndims(space));
    check(H5Sget_simple_extent_dims(space,dims.data(),nullptr));return dims;
}
inline void slab(hid_t data,hid_t memoryType,const void *values,const std::vector<hsize_t> &offset,
                 const std::vector<hsize_t> &size)
{
    Handle target(H5Dget_space(data),H5Sclose);
    check(H5Sselect_hyperslab(target,H5S_SELECT_SET,offset.data(),nullptr,size.data(),nullptr));
    Handle source(H5Screate_simple(int(size.size()),size.data(),nullptr),H5Sclose);
    check(H5Dwrite(data,memoryType,source,target,H5P_DEFAULT,values));
}
inline void table(hid_t parent,const char *name,const char *columns,const std::vector<double> &values,
                  hsize_t rows,hsize_t cols)
{
    if(exists(parent,name)) check(H5Ldelete(parent,name,H5P_DEFAULT));
    Handle data(dataset(parent,name,H5T_IEEE_F64LE,{rows,cols},{128,cols},
                        {H5S_UNLIMITED,cols}),H5Dclose);
    attribute(data,"columns",columns);
    if(rows) check(H5Dwrite(data,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,values.data()));
}
inline void append(hid_t parent,const char *name,const char *columns,const std::vector<double> &row)
{
    if(!exists(parent,name)) table(parent,name,columns,{},0,row.size());
    Handle data(H5Dopen2(parent,name,H5P_DEFAULT),H5Dclose);
    auto dims=dimensions(data);if(dims.size()!=2 || dims[1]!=row.size()) throw std::runtime_error("HDF5 table schema mismatch");
    const hsize_t old=dims[0];++dims[0];check(H5Dset_extent(data,dims.data()));
    slab(data,H5T_NATIVE_DOUBLE,row.data(),{old,0},{1,row.size()});
}
// Binary-search a monotonic step column without loading a full time series.
inline void truncate(hid_t parent,const char *name,uint64_t step)
{
    if(!exists(parent,name)) return;
    Handle data(H5Dopen2(parent,name,H5P_DEFAULT),H5Dclose);auto dims=dimensions(data);
    if(dims.size()!=2 || dims[1]==0) throw std::runtime_error("invalid HDF5 time-series shape");
    hsize_t lo=0,hi=dims[0];
    while(lo<hi)
    {
        const hsize_t mid=lo+(hi-lo)/2;double value;
        Handle space(H5Dget_space(data),H5Sclose);hsize_t offset[]={mid,0},size[]={1,1};
        check(H5Sselect_hyperslab(space,H5S_SELECT_SET,offset,nullptr,size,nullptr));
        Handle mem(H5Screate_simple(2,size,nullptr),H5Sclose);
        check(H5Dread(data,H5T_NATIVE_DOUBLE,mem,space,H5P_DEFAULT,&value));
        if(value<=step) lo=mid+1;else hi=mid;
    }
    dims[0]=lo;check(H5Dset_extent(data,dims.data()));
}
}
