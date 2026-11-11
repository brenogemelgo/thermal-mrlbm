#pragma once

#include "D3Q7.cuh"
#include "deviceFunctions.cuh"
#include "reconstruction.cuh"
#include "hdf5Output.cuh"
#include "hostMomentIO.cuh"
#include <memory>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace snapshot
{
inline std::string &mode();
inline unsigned &workers();
}

static inline void outputCheckCuda(
    const cudaError_t err,
    const char *call)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA output error: " << call << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline const char *outputMomentName(const natural_t field)
{
    switch (field)
    {
    case RHO:
        return "rho";
    case UX:
        return "ux";
    case UY:
        return "uy";
    case UZ:
        return "uz";
    case PXX:
        return "pxx";
    case PYY:
        return "pyy";
    case PZZ:
        return "pzz";
    case PXY:
        return "pxy";
    case PXZ:
        return "pxz";
    case PYZ:
        return "pyz";
    case TEMP:
        return "temperature";
    case TUX:
        return "thermal_flux_x";
    case TUY:
        return "thermal_flux_y";
    case TUZ:
        return "thermal_flux_z";
    default:
        return "unknown";
    }
}

static inline std::string outputStepName(const natural_t step)
{
    std::ostringstream name;
    name << "step_" << std::setw(9) << std::setfill('0') << step;
    return name.str();
}

static inline std::string outputCaseFolderName()
{
    std::string name(ActiveCase::NAME);
    for (char &c : name)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return name;
}

static inline std::string &simulationRunId()
{
    static std::string runId("000");
    return runId;
}

static inline void setSimulationRunId(
    const std::string &runId)
{
    simulationRunId() = runId;
}

static inline std::filesystem::path &simulationOutputRoot()
{
    static std::filesystem::path root("output");
    return root;
}

static inline std::filesystem::path getSimulationOutputDirectory()
{
    return simulationOutputRoot() / outputCaseFolderName() / simulationRunId();
}

// Empty means the legacy co-located layout, or an existing per-run data link.
static inline std::filesystem::path &simulationHdf5Root()
{
    static std::filesystem::path root;
    return root;
}
static inline std::filesystem::path getSimulationHdf5Directory()
{
    if (!simulationHdf5Root().empty())
        return simulationHdf5Root() / outputCaseFolderName() / simulationRunId();
    const auto link = getSimulationOutputDirectory() / "hdf5";
    return std::filesystem::is_symlink(link) || std::filesystem::is_directory(link)
        ? link : getSimulationOutputDirectory();
}

struct OutputSchedule
{
    natural_t slices = 0, fields = 0, profiles = 0, progress = 1000, start = 0, keepCheckpoints = 0;
    natural_t checkpoints = STAMP, end = NSTEPS;
    bool sliceVelocity = false;
};
static inline OutputSchedule &outputSchedule()
{
    static OutputSchedule value;
    return value;
}
static inline std::filesystem::path statisticsPath() { return getSimulationHdf5Directory() / "statistics.h5"; }
static inline std::filesystem::path getMetadataPath() { return getSimulationHdf5Directory() / "metadata.h5"; }
static inline std::filesystem::path canonicalOutputPath(const std::filesystem::path &path)
{
    return std::filesystem::weakly_canonical(std::filesystem::absolute(path));
}
static inline void createOutputDirectories()
{
    const auto control = canonicalOutputPath(getSimulationOutputDirectory());
    const auto data = canonicalOutputPath(getSimulationHdf5Directory());
    std::filesystem::create_directories(control);
    std::filesystem::create_directories(data);
    if (control != data)
    {
        const auto link = control / "hdf5";
        if (std::filesystem::exists(link) || std::filesystem::is_symlink(link))
        {
            if (!std::filesystem::is_symlink(link) || canonicalOutputPath(link) != data)
                throw std::runtime_error("run already points to a different HDF5 location");
        }
        else std::filesystem::create_directory_symlink(data, link);
    }
}

static inline void writeMetadata(
    const uint64_t timeAverageCount = 0u)
{
    createOutputDirectories();

    std::ostringstream out;

    out << std::boolalpha << std::setprecision(17);
    out << "caseName = " << ActiveCase::NAME << '\n';
    out << "case_name = " << ActiveCase::NAME << '\n';
    out << "NX = " << NX << '\n';
    out << "NY = " << NY << '\n';
    out << "NZ = " << NZ << '\n';
    out << "real_bytes = " << sizeof(real_t) << '\n';
    out << "NSTEPS = " << NSTEPS << '\n';
    out << "STAMP = " << STAMP << '\n';
    out << '\n';

    out << "NU = " << static_cast<double>(VISCOSITY) << '\n';
    out << "ALPHA = " << static_cast<double>(THERMAL_DIFFUSIVITY) << '\n';
    out << "KAPPA = " << static_cast<double>(THERMAL_DIFFUSIVITY) << '\n';
    out << "PR = " << static_cast<double>(PRANDTL) << '\n';
    out << "RHO0 = " << static_cast<double>(ActiveCase::RHO0) << '\n';
    out << "moment_format = neq_post_v2" << '\n';
    out << "velocity_convention = v_post_equals_u_plus_half_acceleration" << '\n';
    out << "REGULARIZATION_ORDER = " << RR_ORDER << '\n';
    out << "TAU_SHEAR = " << TAU_SHEAR << '\n';
    out << "TAU_THERMAL = " << TAU_THERMAL << '\n';
    out << '\n';

    out << "ENABLE_BUOYANCY = " << ENABLE_BUOYANCY << '\n';
    out << "ENABLE_INLET = " << ENABLE_INLET << '\n';
    out << "ENABLE_WALLS = " << ENABLE_WALLS << '\n';
    out << "ENABLE_OUTFLOW = " << ENABLE_OUTFLOW << '\n';
    out << '\n';

    out << "PERIODIC_X = " << PERIODIC_X << '\n';
    out << "PERIODIC_Y = " << PERIODIC_Y << '\n';
    out << "PERIODIC_Z = " << PERIODIC_Z << '\n';
    out << '\n';

    out << "GRAVITY_X = " << static_cast<double>(THERMAL_GRAVITY_X) << '\n';
    out << "GRAVITY_Y = " << static_cast<double>(THERMAL_GRAVITY_Y) << '\n';
    out << "GRAVITY_Z = " << static_cast<double>(THERMAL_GRAVITY_Z) << '\n';
    out << '\n';

    out << "THOT = " << static_cast<double>(T_HOT) << '\n';
    out << "TCOLD = " << static_cast<double>(T_COLD) << '\n';
    out << "T_REF = " << static_cast<double>(T_REF) << '\n';
    out << "GBETA = " << static_cast<double>(BETA_THERMAL) << '\n';
#if defined(RBC_VALIDATION_CASE)
    out << "validation_case = V" << RBC_VALIDATION_CASE << '\n';
    out << "validation_formula_version = " << ActiveCase::FORMULA_VERSION << '\n';
    out << "validation_mode_amplitude = " << ActiveCase::MODE_AMPLITUDE << '\n';
    out << "validation_peclet = " << ActiveCase::PECLET << '\n';
    out << "validation_channel_re = " << ActiveCase::CHANNEL_RE << '\n';
    out << "validation_time_scale = diffusive\n";
    out << "validation_source_scheme = trapezoidal_zeroth_moment\n";
    out << "validation_velocity = " << (RBC_VALIDATION_CASE == 2 ? "prescribed" : "solved") << '\n';
#endif

#if defined(CASE_RAYLEIGH_BENARD)
    out << "timeAverageEnabled = " << ActiveCase::timeAverageEnabled << '\n';
    out << "timeAverageStart = " << ActiveCase::timeAverageStart << '\n';
    out << "timeAverageStride = " << ActiveCase::timeAverageStride << '\n';
    out << "timeAverageCount = " << timeAverageCount << '\n';
    out << '\n';
    out << "DELTA_T = " << static_cast<double>(ActiveCase::DELTA_T) << '\n';
    out << "H = " << static_cast<double>(ActiveCase::H) << '\n';
    out << "verticalDirection = " << ActiveCase::verticalDirection << '\n';
    out << "ASPECT_RATIO = " << static_cast<double>(ActiveCase::ASPECT_RATIO) << '\n';
    out << "T0 = " << static_cast<double>(ActiveCase::T0) << '\n';
    out << "U_CHAR = " << static_cast<double>(ActiveCase::U_CHAR) << '\n';
    out << "U_FREEFALL = " << static_cast<double>(ActiveCase::U_FREEFALL) << '\n';
    out << "CONDUCTION_ONLY = " << ActiveCase::CONDUCTION_ONLY << '\n';
    out << "PERTURBATION_AMPLITUDE = " << static_cast<double>(ActiveCase::PERTURBATION_AMPLITUDE) << '\n';
    out << "RE = " << static_cast<double>(ActiveCase::RE) << '\n';
    out << "RA = " << static_cast<double>(ActiveCase::RA) << '\n';
    out << "RBC_CASE = " << ActiveCase::CASE_ID << '\n';
    out << "T_FREEFALL = " << ActiveCase::T_FREEFALL << '\n';
    out << "TRANSIENT_FF = " << ActiveCase::TRANSIENT_FF << '\n';
    out << "AVERAGING_FF = " << ActiveCase::AVERAGING_FF << '\n';
    out << "wall_scheme = on_node_force_corrected_NEE2" << '\n';
#elif defined(CASE_JET)
    out << "timeAverageEnabled = " << ActiveCase::timeAverageEnabled << '\n';
    out << "timeAverageStart = " << ActiveCase::timeAverageStart << '\n';
    out << "timeAverageStride = " << ActiveCase::timeAverageStride << '\n';
    out << "timeAverageCount = " << timeAverageCount << '\n';
    out << '\n';
    out << "U_JET = " << static_cast<double>(ActiveCase::U_JET) << '\n';
    out << "D_JET = " << static_cast<double>(ActiveCase::D_JET) << '\n';
    out << "RE = " << static_cast<double>(ActiveCase::RE) << '\n';
    out << "RI = " << static_cast<double>(ActiveCase::RI) << '\n';
    out << "T_INLET = " << static_cast<double>(ActiveCase::T_INLET) << '\n';
    out << "T_AMBIENT = " << static_cast<double>(ActiveCase::T_AMBIENT) << '\n';
    out << "DELTA_T = " << static_cast<double>(ActiveCase::DELTA_T) << '\n';
#endif
    out << "output_schema = thermal-mrlbm-hdf5-v1\n";
    out << "hdf5_directory = " << canonicalOutputPath(getSimulationHdf5Directory()).string() << '\n';
    out << "vti_enabled = false\n";
    out << "gpu_count = " << (moment_io::distributed() ? moment_io::distributed()->physicalDeviceCount() : 1) << '\n';
    out << "partition_slabs = " << (moment_io::distributed() ? 2 : 1) << '\n';
    out << "partition_axis = " << (moment_io::distributed() ? "z" : "none") << '\n';
    out << "halo_transport = " << (moment_io::distributed() ? moment_io::distributed()->transportName() : "none") << '\n';
    if (moment_io::distributed()) out << "gpu_devices = " << moment_io::distributed()->deviceNames() << '\n';
    out << "run_id = " << simulationRunId() << '\n';
    out << "field_velocity_convention = physical_u\n";
    out << "field_fields = temperature\n";
    out << "field_io = " << snapshot::mode() << '\n';
    out << "io_workers = " << snapshot::workers() << '\n';
    out << "slice_every = " << outputSchedule().slices << '\n';
    out << "field_every = " << outputSchedule().fields << '\n';
    out << "profile_every = " << outputSchedule().profiles << '\n';
    out << "progress_every = " << outputSchedule().progress << '\n';
    out << "checkpoint_every = " << outputSchedule().checkpoints << '\n';
    out << "requested_end_step = " << outputSchedule().end << '\n';
    out << "output_start = " << outputSchedule().start << '\n';
    out << "keep_checkpoints = " << outputSchedule().keepCheckpoints << '\n';
    out << "slice_fields = " << (outputSchedule().sliceVelocity ? "temperature,uz" : "temperature") << '\n';
    out << "compression = " << h5out::compression() << '\n';
    auto temp = getMetadataPath();
    temp += ".tmp";
    {
        h5out::Handle file(H5Fcreate(temp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        h5out::Handle group(h5out::group(file, "metadata"), H5Gclose);
        std::istringstream lines(out.str());
        std::string line;
        auto trim = [](std::string s)
        {const auto a=s.find_first_not_of(" ");return a==s.npos ? std::string() : s.substr(a,s.find_last_not_of(" ")-a+1); };
        while (std::getline(lines, line))
        {
            const auto eq = line.find('=');
            if (eq == line.npos)
                continue;
            h5out::attribute(group, trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
        }
        h5out::flush(file);
    }
    std::filesystem::rename(temp, getMetadataPath());
}

// One XY slab (or one XZ slice), all X accesses contiguous. Output velocities
// use the same buffer-time force subtraction as the production statistics.
__global__ void extractOutputPlane(const real_t *m, real_t *packed, natural_t index, bool xz, natural_t step)
{
    const natural_t q = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t plane = xz ? size_t(NX) * NZ : size_t(NX) * NY;
    if (q >= plane)
        return;
    const natural_t x = q % NX, y = xz ? index : q / NX, z = xz ? q / NX : index;
    const natural_t i = global3(x, y, z);
    const real_t t = loadMoment(m, i, TEMP);
    real_t a[3];
    nodeAcceleration(t, x, y, z, step, a);
    packed[q] = t;
    for (int f = 0; f < 3; ++f)
        packed[size_t(f + 1) * plane + q] = loadMoment(m, i, UX + f) - a[f] / 2;
}
struct PlaneFieldOutput
{
    real_t *device = nullptr;
    std::vector<real_t> host;
    void allocate(bool slice)
    {
        if (device)
            return;
        host.resize(4 * size_t(NX) * std::max(NY, NZ));
        if (slice) outputCheckCuda(cudaMalloc(&device, host.size() * sizeof(real_t)), "output slab allocation");
    }
    ~PlaneFieldOutput()
    {
        if (device)
            cudaFree(device);
    }
    void save(const real_t *m, natural_t step, bool slice)
    {
        if (moment_io::distributed())
            host.resize(4 * size_t(NX) * std::max(NY, NZ));
        else allocate(slice);
        const auto dir = getSimulationHdf5Directory() / (slice ? "slices" : "fields");
        std::filesystem::create_directories(dir);
        const auto path = dir / (outputStepName(step) + ".h5");
        auto tmp = path;
        tmp += ".tmp";
        const hid_t memType = sizeof(real_t) == 4 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
        const hid_t diskType = sizeof(real_t) == 4 ? H5T_IEEE_F32LE : H5T_IEEE_F64LE;
        {
            h5out::Handle file(H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
            h5out::attribute(file, "step", std::to_string(step));
            h5out::attribute(file, "run_id", simulationRunId());
            h5out::attribute(file, "velocity_convention", "physical_u");
            h5out::attribute(file, "axis_order", slice ? "zx" : "zyx");
            h5out::attribute(file, "lattice_spacing", "1");
            h5out::attribute(file, "wall_nodes", "0,NZ-1");
            if (slice)
                h5out::attribute(file, "y_index", std::to_string(NY / 2));
            h5out::Handle group(h5out::group(file, "fields"), H5Gclose);
            const char *names[] = {"temperature", "ux", "uy", "uz"};
            std::vector<std::unique_ptr<h5out::Handle>> datasets;
            const std::vector<hsize_t> dims = slice ? std::vector<hsize_t>{NZ, NX} : std::vector<hsize_t>{NZ, NY, NX};
            const std::vector<hsize_t> chunks = slice ? std::vector<hsize_t>{std::min(64u, NZ), std::min(128u, NX)} : std::vector<hsize_t>{1, std::min(64u, NY), std::min(128u, NX)};
            for (int f = 0; f < 4; ++f)
                if (f == 0 || (slice && outputSchedule().sliceVelocity && f == 3))
                    datasets.emplace_back(new h5out::Handle(h5out::dataset(group, names[f], diskType, dims, chunks), H5Dclose));
            const size_t plane = size_t(NX) * (slice ? NZ : NY);
            for (natural_t z = 0; z < (slice ? 1u : NZ); ++z)
            {
                if (!slice)
                {
                    moment_io::download(host.data(), m, TEMP, z * STRIDE, STRIDE);
                }
                else if (moment_io::distributed())
                {
                    moment_io::distributed()->extract(host.data(), m, slice ? NY / 2 : z, slice, step);
                }
                else
                {
                    extractOutputPlane<<<(plane + 255) / 256, 256>>>(m, device, slice ? NY / 2 : z, slice, step);
                    outputCheckCuda(cudaGetLastError(), "extract output plane");
                    outputCheckCuda(cudaMemcpy(host.data(), device, 4 * plane * sizeof(real_t), cudaMemcpyDeviceToHost), "copy output plane");
                }
                int d = 0;
                for (int f = 0; f < 4; ++f)
                    if (f == 0 || (slice && outputSchedule().sliceVelocity && f == 3))
                    {
                        if (slice)
                            h5out::slab(*datasets[d++], memType, host.data() + f * plane, {0, 0}, {NZ, NX});
                        else
                            h5out::slab(*datasets[d++], memType, host.data() + f * plane, {z, 0, 0}, {1, NY, NX});
                    }
            }
            h5out::flush(file);
        }
        std::filesystem::rename(tmp, path);
    }
};

#include "temperatureOutput.cuh"

struct FieldOutput
{
    PlaneFieldOutput planes;
    snapshot::TemperatureOutput temperature;
    void save(const real_t *moments, natural_t step, bool slice)
    {
        if (slice || snapshot::mode() == "serial") planes.save(moments, step, slice);
        else temperature.save(moments, step);
    }
    void poll() {temperature.poll();}
    void drain() {temperature.drain();}
    double backgroundSeconds() const {return temperature.backgroundSeconds();}
};
static inline void writeScalarOutput(const real_t *deviceField, const char *name)
{
    const auto path = getSimulationHdf5Directory() / (std::string(name) + ".h5");
    auto tmp = path;
    tmp += ".tmp";
    {
        h5out::Handle file(H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        const hid_t mem = sizeof(real_t) == 4 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
        const hid_t disk = sizeof(real_t) == 4 ? H5T_IEEE_F32LE : H5T_IEEE_F64LE;
        h5out::Handle data(h5out::dataset(file, name, disk, {NZ, NY, NX}, {1, std::min(64u, NY), std::min(128u, NX)}), H5Dclose);
        std::vector<real_t> plane(size_t(NX) * NY);
        for (natural_t z = 0; z < NZ; ++z)
        {
            outputCheckCuda(cudaMemcpy(plane.data(), deviceField + size_t(z) * plane.size(), plane.size() * sizeof(real_t), cudaMemcpyDeviceToHost), "scalar plane");
            h5out::slab(data, mem, plane.data(), {z, 0, 0}, {1, NY, NX});
        }
        h5out::flush(file);
    }
    std::filesystem::rename(tmp, path);
}
