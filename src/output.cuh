#pragma once

#include "D3Q7.cuh"
#include "deviceFunctions.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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
    case TFX:
        return "thermal_flux_x";
    case TFY:
        return "thermal_flux_y";
    case TFZ:
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

static inline std::filesystem::path &simulationOutputDirectory()
{
    static std::filesystem::path dir("output/default");
    return dir;
}

static inline void setSimulationOutputDirectory(
    const std::filesystem::path &dir)
{
    simulationOutputDirectory() = dir;
}

static inline const std::filesystem::path &getSimulationOutputDirectory()
{
    return simulationOutputDirectory();
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
    const std::filesystem::path dir = getSimulationOutputDirectory();
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
        std::cerr << "No checkpoint found in " << getSimulationOutputDirectory() << std::endl;
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

struct ThermalDiagnostics
{
    double minTemperature = 0.0;
    double maxTemperature = 0.0;
    double sumTemperature = 0.0;
    double minDensity = 0.0;
    double maxDensity = 0.0;
    double maxVelocity = 0.0;
    double maxAxialVelocity = 0.0;
    double maxHydroForce = 0.0;
    double inletMassFlux = 0.0;
    double inletThermalFlux = 0.0;
    natural_t nonFiniteTemperatureCells = 0;
    natural_t nonFiniteVelocityCells = 0;
};

static inline bool isJetInletHost(
    const natural_t x,
    const natural_t y,
    const natural_t z)
{
    if (z != 0)
    {
        return false;
    }

    constexpr double cx0 = 0.5 * static_cast<double>(NX - 1);
    constexpr double cy0 = 0.5 * static_cast<double>(NY - 1);

    constexpr double radius2 = static_cast<double>(RADIUS) * static_cast<double>(RADIUS);

    const double dx = static_cast<double>(x) - cx0;
    const double dy = static_cast<double>(y) - cy0;

    return dx * dx + dy * dy <= radius2;
}

static inline void copyTemperatureToHost(
    const real_t *deviceMoments,
    std::vector<real_t> &temperature)
{
    temperature.assign(CELLS, static_cast<real_t>(0));
    outputCheckCuda(
        cudaMemcpy(temperature.data(), deviceMoments + CELLS * TEMP, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy temperature field");
}

static inline ThermalDiagnostics computeThermalDiagnostics(
    const real_t *deviceMoments)
{
    std::vector<real_t> temperature;
    copyTemperatureToHost(deviceMoments, temperature);

    ThermalDiagnostics diagnostics;
    diagnostics.minTemperature = std::numeric_limits<double>::infinity();
    diagnostics.maxTemperature = -std::numeric_limits<double>::infinity();
    diagnostics.minDensity = std::numeric_limits<double>::infinity();
    diagnostics.maxDensity = -std::numeric_limits<double>::infinity();
    diagnostics.maxAxialVelocity = -std::numeric_limits<double>::infinity();

    for (const real_t value : temperature)
    {
        const double t = static_cast<double>(value);
        if (!std::isfinite(t))
        {
            ++diagnostics.nonFiniteTemperatureCells;
            continue;
        }

        diagnostics.minTemperature = std::min(diagnostics.minTemperature, t);
        diagnostics.maxTemperature = std::max(diagnostics.maxTemperature, t);
        diagnostics.sumTemperature += t;

        const double forceX =
            static_cast<double>(BETA_THERMAL) *
            static_cast<double>(THERMAL_FORCE_SCALE) *
            (t - static_cast<double>(T_REF)) *
            static_cast<double>(THERMAL_GRAVITY_X);
        const double forceY =
            static_cast<double>(BETA_THERMAL) *
            static_cast<double>(THERMAL_FORCE_SCALE) *
            (t - static_cast<double>(T_REF)) *
            static_cast<double>(THERMAL_GRAVITY_Y);
        const double forceZ =
            static_cast<double>(BETA_THERMAL) *
            static_cast<double>(THERMAL_FORCE_SCALE) *
            (t - static_cast<double>(T_REF)) *
            static_cast<double>(THERMAL_GRAVITY_Z);
        const double forceNorm = std::sqrt(forceX * forceX + forceY * forceY + forceZ * forceZ);
        diagnostics.maxHydroForce = std::max(diagnostics.maxHydroForce, forceNorm);
    }

    std::vector<real_t> rho(CELLS);
    std::vector<real_t> ux(CELLS);
    std::vector<real_t> uy(CELLS);
    std::vector<real_t> uz(CELLS);

    outputCheckCuda(
        cudaMemcpy(rho.data(), deviceMoments + CELLS * RHO, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy rho diagnostics");
    outputCheckCuda(
        cudaMemcpy(ux.data(), deviceMoments + CELLS * UX, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy ux diagnostics");
    outputCheckCuda(
        cudaMemcpy(uy.data(), deviceMoments + CELLS * UY, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy uy diagnostics");
    outputCheckCuda(
        cudaMemcpy(uz.data(), deviceMoments + CELLS * UZ, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy uz diagnostics");

    const double invScaleI = 1.0 / static_cast<double>(VelocitySet::scaleI());
    for (natural_t idx = 0; idx < CELLS; ++idx)
    {
        const double density = static_cast<double>(rho[idx]);
        if (std::isfinite(density))
        {
            diagnostics.minDensity = std::min(diagnostics.minDensity, density);
            diagnostics.maxDensity = std::max(diagnostics.maxDensity, density);
        }

        const double vx = static_cast<double>(ux[idx]) * invScaleI;
        const double vy = static_cast<double>(uy[idx]) * invScaleI;
        const double vz = static_cast<double>(uz[idx]) * invScaleI;
        if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz))
        {
            ++diagnostics.nonFiniteVelocityCells;
            continue;
        }

        diagnostics.maxVelocity = std::max(diagnostics.maxVelocity, std::sqrt(vx * vx + vy * vy + vz * vz));
        diagnostics.maxAxialVelocity = std::max(diagnostics.maxAxialVelocity, vz);

        const natural_t z = idx / STRIDE;
        const natural_t rem = idx - z * STRIDE;
        const natural_t y = rem / NX;
        const natural_t x = rem - y * NX;
        const double temperatureValue = static_cast<double>(temperature[idx]);
        if (isJetInletHost(x, y, z) &&
            std::isfinite(density) &&
            std::isfinite(temperatureValue))
        {
            diagnostics.inletMassFlux += density * vz;
            diagnostics.inletThermalFlux += temperatureValue * vz;
        }
    }

    if (diagnostics.nonFiniteTemperatureCells == CELLS)
    {
        diagnostics.minTemperature = std::numeric_limits<double>::quiet_NaN();
        diagnostics.maxTemperature = std::numeric_limits<double>::quiet_NaN();
        diagnostics.sumTemperature = std::numeric_limits<double>::quiet_NaN();
        diagnostics.maxHydroForce = std::numeric_limits<double>::quiet_NaN();
    }
    if (!std::isfinite(diagnostics.minDensity))
    {
        diagnostics.minDensity = std::numeric_limits<double>::quiet_NaN();
        diagnostics.maxDensity = std::numeric_limits<double>::quiet_NaN();
    }
    if (!std::isfinite(diagnostics.maxAxialVelocity))
    {
        diagnostics.maxAxialVelocity = std::numeric_limits<double>::quiet_NaN();
    }

    return diagnostics;
}

static inline void printThermalDiagnosticsHeader()
{
    std::cout << "thermal diagnostics:" << std::endl;
    std::cout << "step,minT,maxT,sumT,minRho,maxRho,maxU,maxUz,hydroForceMax,badT,badU,inletMassFlux,inletThermalFlux" << std::endl;
}

static inline void printThermalDiagnosticsLine(
    const natural_t step,
    const ThermalDiagnostics &diagnostics)
{
    std::cout << std::setprecision(10)
              << step << ','
              << diagnostics.minTemperature << ','
              << diagnostics.maxTemperature << ','
              << diagnostics.sumTemperature << ','
              << diagnostics.minDensity << ','
              << diagnostics.maxDensity << ','
              << diagnostics.maxVelocity << ','
              << diagnostics.maxAxialVelocity << ','
              << diagnostics.maxHydroForce << ','
              << diagnostics.nonFiniteTemperatureCells << ','
              << diagnostics.nonFiniteVelocityCells << ','
              << diagnostics.inletMassFlux << ','
              << diagnostics.inletThermalFlux << std::endl;
}

static inline void writeOutput(
    const real_t *deviceMoments,
    const natural_t step)
{
    const std::filesystem::path dir = getSimulationOutputDirectory();
    std::filesystem::create_directories(dir);

    const std::string base = outputStepName(step);
    const std::filesystem::path binaryPath = dir / (base + ".bin");
    const std::filesystem::path vtiPath = dir / (base + ".vti");

    writeBinary(deviceMoments, binaryPath);
    writeVti(binaryPath, vtiPath);
}
