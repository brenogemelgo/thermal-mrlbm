#pragma once

#include "D3Q7.cuh"
#include "deviceFunctions.cuh"

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

static inline std::filesystem::path &simulationOutputRoot()
{
    static std::filesystem::path root("output");
    return root;
}

static inline std::filesystem::path getSimulationOutputDirectory()
{
    return simulationOutputRoot() / outputCaseFolderName() / simulationRunId();
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
    const uint64_t timeAverageCount = 0u)
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
    out << "validation_velocity = " << (RBC_VALIDATION_CASE==2 ? "prescribed" : "solved") << '\n';
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
