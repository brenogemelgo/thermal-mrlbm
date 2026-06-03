#pragma once

#include "D3Q7.cuh"
#include "deviceFunctions.cuh"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

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
    case MXX:
        return "mxx";
    case MYY:
        return "myy";
    case MZZ:
        return "mzz";
    case MXY:
        return "mxy";
    case MXZ:
        return "mxz";
    case MYZ:
        return "myz";
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

static inline const char *outputVtkRealType()
{
    if constexpr (std::is_same_v<real_t, float>)
    {
        return "Float32";
    }
    else
    {
        return "Float64";
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

static inline std::filesystem::path getSimulationOutputDirectory()
{
    return std::filesystem::path("output") / outputCaseFolderName() / simulationRunId();
}

static inline std::filesystem::path getBinaryOutputDirectory()
{
    return getSimulationOutputDirectory() / "binaries";
}

static inline std::filesystem::path getVtiOutputDirectory()
{
    return getSimulationOutputDirectory() / "vtis";
}

static inline std::filesystem::path getMetadataPath()
{
    return getSimulationOutputDirectory() / "metadata.txt";
}

static inline void createOutputDirectories()
{
    std::filesystem::create_directories(getBinaryOutputDirectory());
    std::filesystem::create_directories(getVtiOutputDirectory());
}

static inline void writeMetadata(
    const natural_t timeAverageCount = 0u)
{
    createOutputDirectories();

    std::ofstream out(getMetadataPath());
    if (!out)
    {
        std::cerr << "Could not open metadata output: " << getMetadataPath() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    out << std::boolalpha << std::setprecision(17);
    out << "caseName = " << ActiveCase::NAME << '\n';
    out << "NX = " << NX << '\n';
    out << "NY = " << NY << '\n';
    out << "NZ = " << NZ << '\n';
    out << "NSTEPS = " << NSTEPS << '\n';
    out << "STAMP = " << STAMP << '\n';
    out << '\n';

    out << "NU = " << static_cast<double>(VISCOSITY) << '\n';
    out << "ALPHA = " << static_cast<double>(THERMAL_DIFFUSIVITY) << '\n';
    out << "PR = " << static_cast<double>(PRANDTL) << '\n';
    out << "RHO0 = " << static_cast<double>(ActiveCase::RHO0) << '\n';
    out << "velocityScaleI = " << static_cast<double>(VelocitySet::scaleI()) << '\n';
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

#if defined(CASE_FOURIER)
    out << "timeAverageEnabled = false" << '\n';
    out << "timeAverageCount = 0" << '\n';
    out << '\n';
    out << "T0 = " << static_cast<double>(ActiveCase::T0) << '\n';
    out << "AMP_T = " << static_cast<double>(ActiveCase::AMP_T) << '\n';
    out << "UX0 = " << static_cast<double>(ActiveCase::UX0) << '\n';
    out << "UY0 = " << static_cast<double>(ActiveCase::UY0) << '\n';
    out << "UZ0 = " << static_cast<double>(ActiveCase::UZ0) << '\n';
    out << "MODE_X = " << ActiveCase::MODE_X << '\n';
    out << "MODE_Y = " << ActiveCase::MODE_Y << '\n';
    out << "MODE_Z = " << ActiveCase::MODE_Z << '\n';
    out << "KX = " << static_cast<double>(ActiveCase::KX) << '\n';
    out << "KY = " << static_cast<double>(ActiveCase::KY) << '\n';
    out << "KZ = " << static_cast<double>(ActiveCase::KZ) << '\n';
    out << "K2 = " << static_cast<double>(ActiveCase::K2) << '\n';
    out << "DECAY_RATE = " << static_cast<double>(ActiveCase::DECAY_RATE) << '\n';
#elif defined(CASE_CAVITY)
    out << "timeAverageEnabled = false" << '\n';
    out << "timeAverageCount = 0" << '\n';
    out << '\n';
    out << "RA = " << static_cast<double>(ActiveCase::RA) << '\n';
    out << "DELTA_T = " << static_cast<double>(ActiveCase::DELTA_T) << '\n';
    out << "L_CAVITY = " << static_cast<double>(ActiveCase::L_CAVITY) << '\n';
    out << "UFREEFALL_TARGET = " << static_cast<double>(ActiveCase::UFREEFALL_TARGET) << '\n';
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
}

static inline void initializeOutputLayout()
{
    createOutputDirectories();
    writeMetadata();
}

static inline void writeBinary(
    const real_t *deviceMoments,
    const std::filesystem::path &path)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        std::cerr << "Could not open binary output: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<real_t> fieldData(CELLS);

    for (natural_t field = 0; field < NUM_MOMENTS; ++field)
    {
        outputCheckCuda(
            cudaMemcpy(fieldData.data(), deviceMoments + CELLS * field, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy binary field");

        out.write(reinterpret_cast<const char *>(fieldData.data()), static_cast<std::streamsize>(CELLS * sizeof(real_t)));
        if (!out)
        {
            std::cerr << "Could not write binary output: " << path << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

static inline bool parseOutputBinaryStep(
    const std::filesystem::path &path,
    natural_t &step)
{
    if (path.extension() != ".bin")
    {
        return false;
    }

    const std::string stem = path.stem().string();
    constexpr const char *prefix = "step_";
    constexpr std::size_t prefixSize = 5;

    if (stem.size() <= prefixSize || stem.compare(0, prefixSize, prefix) != 0)
    {
        return false;
    }

    natural_t value = 0;
    for (std::size_t i = prefixSize; i < stem.size(); ++i)
    {
        const char c = stem[i];
        if (c < '0' || c > '9')
        {
            return false;
        }
        value = value * static_cast<natural_t>(10) + static_cast<natural_t>(c - '0');
    }

    step = value;
    return true;
}

static inline bool findLatestOutputBinary(
    std::filesystem::path &binaryPath,
    natural_t &step)
{
    const std::filesystem::path dir = getBinaryOutputDirectory();
    if (!std::filesystem::exists(dir))
    {
        return false;
    }

    bool found = false;
    natural_t latestStep = 0;
    std::filesystem::path latestPath;

    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        natural_t candidateStep = 0;
        if (!parseOutputBinaryStep(entry.path(), candidateStep))
        {
            continue;
        }

        if (!found || candidateStep > latestStep)
        {
            found = true;
            latestStep = candidateStep;
            latestPath = entry.path();
        }
    }

    if (found)
    {
        step = latestStep;
        binaryPath = latestPath;
    }

    return found;
}

static inline void readBinary(
    real_t *deviceMoments,
    const std::filesystem::path &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::cerr << "Could not open checkpoint input: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<real_t> fieldData(CELLS);

    for (natural_t field = 0; field < NUM_MOMENTS; ++field)
    {
        in.read(reinterpret_cast<char *>(fieldData.data()), static_cast<std::streamsize>(CELLS * sizeof(real_t)));
        if (!in)
        {
            std::cerr << "Could not read checkpoint field " << outputMomentName(field) << ": " << path << std::endl;
            std::exit(EXIT_FAILURE);
        }

        outputCheckCuda(
            cudaMemcpy(deviceMoments + CELLS * field, fieldData.data(), CELLS * sizeof(real_t), cudaMemcpyHostToDevice),
            "cudaMemcpy checkpoint field");
    }
}

static inline natural_t loadLatestCheckpoint(
    real_t *moments,
    real_t *dbuffer)
{
    std::filesystem::path binaryPath;
    natural_t step = 0;
    if (!findLatestOutputBinary(binaryPath, step))
    {
        std::cerr << "No checkpoint found in " << getBinaryOutputDirectory() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    readBinary(moments, binaryPath);
    outputCheckCuda(
        cudaMemcpy(dbuffer, moments, static_cast<size_t>(NUM_MOMENTS) * static_cast<size_t>(CELLS) * sizeof(real_t), cudaMemcpyDeviceToDevice),
        "cudaMemcpy checkpoint buffer");

    std::cout << "loaded checkpoint " << binaryPath << " at step " << step << std::endl;
    return step;
}

static inline void writeVti(
    const std::filesystem::path &binaryPath,
    const std::filesystem::path &vtiPath)
{
    std::ifstream bin(binaryPath, std::ios::binary);
    if (!bin)
    {
        std::cerr << "Could not open binary input for VTI: " << binaryPath << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::ofstream vti(vtiPath, std::ios::binary);
    if (!vti)
    {
        std::cerr << "Could not open VTI output: " << vtiPath << std::endl;
        std::exit(EXIT_FAILURE);
    }

    constexpr std::uint64_t fieldBytes = static_cast<std::uint64_t>(CELLS) * static_cast<std::uint64_t>(sizeof(real_t));
    std::uint64_t offset = 0;

    vti << "<?xml version=\"1.0\"?>\n";
    vti << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vti << "  <ImageData WholeExtent=\"0 " << (NX - 1) << " 0 " << (NY - 1) << " 0 " << (NZ - 1)
        << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
    vti << "    <Piece Extent=\"0 " << (NX - 1) << " 0 " << (NY - 1) << " 0 " << (NZ - 1) << "\">\n";
    vti << "      <PointData Scalars=\"rho\">\n";

    for (natural_t field = 0; field < NUM_MOMENTS; ++field)
    {
        vti << "        <DataArray type=\"" << outputVtkRealType() << "\" Name=\"" << outputMomentName(field)
            << "\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << offset << "\"/>\n";
        offset += sizeof(std::uint64_t) + fieldBytes;
    }

    vti << "      </PointData>\n";
    vti << "      <CellData/>\n";
    vti << "    </Piece>\n";
    vti << "  </ImageData>\n";
    vti << "  <AppendedData encoding=\"raw\">\n_";

    std::vector<real_t> fieldData(CELLS);
    for (natural_t field = 0; field < NUM_MOMENTS; ++field)
    {
        bin.read(reinterpret_cast<char *>(fieldData.data()), static_cast<std::streamsize>(fieldBytes));
        if (!bin)
        {
            std::cerr << "Could not read binary field for VTI: " << binaryPath << std::endl;
            std::exit(EXIT_FAILURE);
        }

        vti.write(reinterpret_cast<const char *>(&fieldBytes), sizeof(fieldBytes));
        vti.write(reinterpret_cast<const char *>(fieldData.data()), static_cast<std::streamsize>(fieldBytes));
        if (!vti)
        {
            std::cerr << "Could not write VTI output: " << vtiPath << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    vti << "\n  </AppendedData>\n";
    vti << "</VTKFile>\n";
}

static inline void writeScalarBinary(
    const real_t *deviceField,
    const std::filesystem::path &path)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        std::cerr << "Could not open scalar binary output: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<real_t> fieldData(CELLS);
    outputCheckCuda(
        cudaMemcpy(fieldData.data(), deviceField, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy scalar binary field");

    out.write(reinterpret_cast<const char *>(fieldData.data()), static_cast<std::streamsize>(CELLS * sizeof(real_t)));
    if (!out)
    {
        std::cerr << "Could not write scalar binary output: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline void writeScalarVti(
    const std::filesystem::path &binaryPath,
    const std::filesystem::path &vtiPath,
    const char *fieldName)
{
    std::ifstream bin(binaryPath, std::ios::binary);
    if (!bin)
    {
        std::cerr << "Could not open scalar binary input for VTI: " << binaryPath << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::ofstream vti(vtiPath, std::ios::binary);
    if (!vti)
    {
        std::cerr << "Could not open scalar VTI output: " << vtiPath << std::endl;
        std::exit(EXIT_FAILURE);
    }

    constexpr std::uint64_t fieldBytes = static_cast<std::uint64_t>(CELLS) * static_cast<std::uint64_t>(sizeof(real_t));

    vti << "<?xml version=\"1.0\"?>\n";
    vti << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vti << "  <ImageData WholeExtent=\"0 " << (NX - 1) << " 0 " << (NY - 1) << " 0 " << (NZ - 1)
        << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
    vti << "    <Piece Extent=\"0 " << (NX - 1) << " 0 " << (NY - 1) << " 0 " << (NZ - 1) << "\">\n";
    vti << "      <PointData Scalars=\"" << fieldName << "\">\n";
    vti << "        <DataArray type=\"" << outputVtkRealType() << "\" Name=\"" << fieldName
        << "\" NumberOfComponents=\"1\" format=\"appended\" offset=\"0\"/>\n";
    vti << "      </PointData>\n";
    vti << "      <CellData/>\n";
    vti << "    </Piece>\n";
    vti << "  </ImageData>\n";
    vti << "  <AppendedData encoding=\"raw\">\n_";

    std::vector<real_t> fieldData(CELLS);
    bin.read(reinterpret_cast<char *>(fieldData.data()), static_cast<std::streamsize>(fieldBytes));
    if (!bin)
    {
        std::cerr << "Could not read scalar binary field for VTI: " << binaryPath << std::endl;
        std::exit(EXIT_FAILURE);
    }

    vti.write(reinterpret_cast<const char *>(&fieldBytes), sizeof(fieldBytes));
    vti.write(reinterpret_cast<const char *>(fieldData.data()), static_cast<std::streamsize>(fieldBytes));
    if (!vti)
    {
        std::cerr << "Could not write scalar VTI output: " << vtiPath << std::endl;
        std::exit(EXIT_FAILURE);
    }

    vti << "\n  </AppendedData>\n";
    vti << "</VTKFile>\n";
}

static inline void writeScalarOutput(
    const real_t *deviceField,
    const char *fieldName)
{
    createOutputDirectories();

    const std::filesystem::path binaryPath = getBinaryOutputDirectory() / (std::string(fieldName) + ".bin");
    const std::filesystem::path vtiPath = getVtiOutputDirectory() / (std::string(fieldName) + ".vti");

    writeScalarBinary(deviceField, binaryPath);
    writeScalarVti(binaryPath, vtiPath, fieldName);
}

static inline void writeOutput(
    const real_t *deviceMoments,
    const natural_t step)
{
    createOutputDirectories();

    const std::string base = outputStepName(step);
    const std::filesystem::path binaryPath = getBinaryOutputDirectory() / (base + ".bin");
    const std::filesystem::path vtiPath = getVtiOutputDirectory() / (base + ".vti");

    writeBinary(deviceMoments, binaryPath);
    writeVti(binaryPath, vtiPath);
}

__global__ void accumulateJetTimeAverage(
    const real_t *__restrict__ moments,
    real_t *__restrict__ temperatureAccum,
    real_t *__restrict__ uzAccum)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
    temperatureAccum[idx] += loadMoment(moments, idx, TEMP);
    uzAccum[idx] += loadMoment(moments, idx, UZ) / VelocitySet::scaleI();
}

__global__ void finalizeJetTimeAverage(
    const real_t *__restrict__ temperatureAccum,
    const real_t *__restrict__ uzAccum,
    real_t *__restrict__ temperatureMean,
    real_t *__restrict__ uzMean,
    const natural_t timeAverageCount)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
    const real_t invCount = timeAverageCount > 0u
                                ? static_cast<real_t>(1) / static_cast<real_t>(timeAverageCount)
                                : static_cast<real_t>(0);

    temperatureMean[idx] = temperatureAccum[idx] * invCount;
    uzMean[idx] = uzAccum[idx] * invCount;
}
