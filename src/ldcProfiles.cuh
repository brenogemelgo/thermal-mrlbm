#pragma once

#include "deviceFunctions.cuh"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

constexpr natural_t LDC_PROFILE_THREADS = 256;
constexpr natural_t LDC_PROFILE_FIELDS = 5;

constexpr natural_t LDC_PROFILE_UX = 0;
constexpr natural_t LDC_PROFILE_UZ = 1;
constexpr natural_t LDC_PROFILE_UX2 = 2;
constexpr natural_t LDC_PROFILE_UZ2 = 3;
constexpr natural_t LDC_PROFILE_UXUZ = 4;

constexpr natural_t LDC_PROFILE_CX_OFFSET = 0;
constexpr natural_t LDC_PROFILE_CY_OFFSET = NX * LDC_PROFILE_FIELDS;
constexpr natural_t LDC_PROFILE_VALUES = (NX + NZ) * LDC_PROFILE_FIELDS;

struct LdcProfileSamples
{
    real_t *deviceSample = nullptr;
    std::vector<real_t> hostSample;
    std::ofstream cxSamplesOutput;
    std::ofstream cySamplesOutput;
    std::ofstream stepsOutput;
    std::filesystem::path profileDir;
    natural_t samples = 0;
};

__device__ [[nodiscard]] static __forceinline__ natural_t ldcProfileIndex(
    const natural_t sample,
    const natural_t offset,
    const natural_t coordinate,
    const natural_t field) noexcept
{
    return sample * LDC_PROFILE_VALUES + offset + coordinate * LDC_PROFILE_FIELDS + field;
}

__device__ static __forceinline__ void ldcAccumulatePoint(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &sumUx,
    real_t &sumUz,
    real_t &sumUx2,
    real_t &sumUz2,
    real_t &sumUxUz,
    natural_t &samples) noexcept
{
    const natural_t idx = global3(x, y, z);
    const real_t invVelocityScale = static_cast<real_t>(1) / VelocitySet::scaleI();
    const real_t ux = loadMoment(moments, idx, UX) * invVelocityScale;
    const real_t uz = loadMoment(moments, idx, UZ) * invVelocityScale;

    sumUx += ux;
    sumUz += uz;
    sumUx2 += ux * ux;
    sumUz2 += uz * uz;
    sumUxUz += ux * uz;
    ++samples;
}

__global__ void writeLdcProfileSampleKernel(
    const real_t *__restrict__ moments,
    real_t *__restrict__ samples,
    const natural_t sample)
{
    const natural_t i = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr natural_t x0 = (NX - 1) / 2;
    constexpr natural_t x1 = NX / 2;
    constexpr natural_t y0 = (NY - 1) / 2;
    constexpr natural_t y1 = NY / 2;
    constexpr natural_t z0 = (NZ - 1) / 2;
    constexpr natural_t z1 = NZ / 2;

    if (i < NX)
    {
        real_t sumUx = static_cast<real_t>(0);
        real_t sumUz = static_cast<real_t>(0);
        real_t sumUx2 = static_cast<real_t>(0);
        real_t sumUz2 = static_cast<real_t>(0);
        real_t sumUxUz = static_cast<real_t>(0);
        natural_t sampleCount = 0;

        ldcAccumulatePoint(moments, i, y0, z0, sumUx, sumUz, sumUx2, sumUz2, sumUxUz, sampleCount);
        if constexpr (z1 != z0)
        {
            ldcAccumulatePoint(moments, i, y0, z1, sumUx, sumUz, sumUx2, sumUz2, sumUxUz, sampleCount);
        }
        if constexpr (y1 != y0)
        {
            ldcAccumulatePoint(moments, i, y1, z0, sumUx, sumUz, sumUx2, sumUz2, sumUxUz, sampleCount);
            if constexpr (z1 != z0)
            {
                ldcAccumulatePoint(moments, i, y1, z1, sumUx, sumUz, sumUx2, sumUz2, sumUxUz, sampleCount);
            }
        }

        const real_t invSamples = static_cast<real_t>(1) / static_cast<real_t>(sampleCount);
        samples[ldcProfileIndex(sample, LDC_PROFILE_CX_OFFSET, i, LDC_PROFILE_UX)] = sumUx * invSamples;
        samples[ldcProfileIndex(sample, LDC_PROFILE_CX_OFFSET, i, LDC_PROFILE_UZ)] = sumUz * invSamples;
        samples[ldcProfileIndex(sample, LDC_PROFILE_CX_OFFSET, i, LDC_PROFILE_UX2)] = sumUx2 * invSamples;
        samples[ldcProfileIndex(sample, LDC_PROFILE_CX_OFFSET, i, LDC_PROFILE_UZ2)] = sumUz2 * invSamples;
        samples[ldcProfileIndex(sample, LDC_PROFILE_CX_OFFSET, i, LDC_PROFILE_UXUZ)] = sumUxUz * invSamples;
    }

    if (i < NZ)
    {
        real_t sumUx = static_cast<real_t>(0);
        real_t sumUz = static_cast<real_t>(0);
        real_t sumUx2 = static_cast<real_t>(0);
        real_t sumUz2 = static_cast<real_t>(0);
        real_t sumUxUz = static_cast<real_t>(0);
        natural_t sampleCount = 0;

        ldcAccumulatePoint(moments, x0, y0, i, sumUx, sumUz, sumUx2, sumUz2, sumUxUz, sampleCount);
        if constexpr (x1 != x0)
        {
            ldcAccumulatePoint(moments, x1, y0, i, sumUx, sumUz, sumUx2, sumUz2, sumUxUz, sampleCount);
        }
        if constexpr (y1 != y0)
        {
            ldcAccumulatePoint(moments, x0, y1, i, sumUx, sumUz, sumUx2, sumUz2, sumUxUz, sampleCount);
            if constexpr (x1 != x0)
            {
                ldcAccumulatePoint(moments, x1, y1, i, sumUx, sumUz, sumUx2, sumUz2, sumUxUz, sampleCount);
            }
        }

        const real_t invSamples = static_cast<real_t>(1) / static_cast<real_t>(sampleCount);
        samples[ldcProfileIndex(sample, LDC_PROFILE_CY_OFFSET, i, LDC_PROFILE_UX)] = sumUx * invSamples;
        samples[ldcProfileIndex(sample, LDC_PROFILE_CY_OFFSET, i, LDC_PROFILE_UZ)] = sumUz * invSamples;
        samples[ldcProfileIndex(sample, LDC_PROFILE_CY_OFFSET, i, LDC_PROFILE_UX2)] = sumUx2 * invSamples;
        samples[ldcProfileIndex(sample, LDC_PROFILE_CY_OFFSET, i, LDC_PROFILE_UZ2)] = sumUz2 * invSamples;
        samples[ldcProfileIndex(sample, LDC_PROFILE_CY_OFFSET, i, LDC_PROFILE_UXUZ)] = sumUxUz * invSamples;
    }
}

static inline double ldcProfileCoordinate(
    const natural_t coordinate,
    const natural_t length)
{
    if (length <= 1)
    {
        return 0.0;
    }

    return 2.0 * static_cast<double>(coordinate) / static_cast<double>(length - 1) - 1.0;
}

static inline void writeLdcProfileCoordinates(
    const std::filesystem::path &path,
    const natural_t length)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "Could not open LDC profile coordinate output: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    out << std::setprecision(17);
    out << "index,s\n";
    for (natural_t coordinate = 0; coordinate < length; ++coordinate)
    {
        out << coordinate << ',' << ldcProfileCoordinate(coordinate, length) << '\n';
    }
}

static inline natural_t countExistingLdcProfileSamples(
    const std::filesystem::path &stepsPath)
{
    std::ifstream in(stepsPath);
    if (!in)
    {
        return 0;
    }

    std::string line;
    natural_t samples = 0;
    bool firstLine = true;
    while (std::getline(in, line))
    {
        if (firstLine)
        {
            firstLine = false;
            continue;
        }
        if (!line.empty())
        {
            ++samples;
        }
    }

    return samples;
}

static inline bool existingLdcProfileSamplesAreCompatible(
    const std::filesystem::path &metadataPath)
{
    std::ifstream in(metadataPath);
    if (!in)
    {
        return false;
    }

    std::string line;
    if (!std::getline(in, line))
    {
        return false;
    }

    while (std::getline(in, line))
    {
        constexpr const char *key = "sample_velocity_scale,";
        constexpr std::size_t keyLength = 22;
        if (line.compare(0, keyLength, key) == 0)
        {
            return line.substr(keyLength) == "1";
        }
    }

    return false;
}

static inline void writeLdcProfileMetadata(
    const std::filesystem::path &profileDir,
    const natural_t samples)
{
    std::ofstream metadata(profileDir / "metadata.csv");
    if (!metadata)
    {
        std::cerr << "Could not open LDC profile metadata output" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    metadata << "key,value\n";
    metadata << "sample_count," << samples << '\n';
    metadata << "cx_length," << NX << '\n';
    metadata << "cy_length," << NZ << '\n';
    metadata << "fields,ux;uz;ux2;uz2;uxuz\n";
    metadata << "real_t_bytes," << sizeof(real_t) << '\n';
    metadata << "sample_velocity_scale,1\n";
    metadata << "u_char," << std::setprecision(17) << static_cast<double>(U_CHAR) << '\n';
    metadata << "coordinate_min,-1\n";
    metadata << "coordinate_max,1\n";
}

static inline cudaError_t initLdcProfileSamples(
    LdcProfileSamples &profiles,
    const std::filesystem::path &outputDir,
    const bool append)
{
    profiles.hostSample.assign(LDC_PROFILE_VALUES, static_cast<real_t>(0));
    profiles.profileDir = outputDir / "profiles";
    std::filesystem::create_directories(profiles.profileDir);

    const std::filesystem::path cxPath = profiles.profileDir / "centerline_cx_samples.bin";
    const std::filesystem::path cyPath = profiles.profileDir / "centerline_cy_samples.bin";
    const std::filesystem::path stepsPath = profiles.profileDir / "sample_steps.csv";
    const std::filesystem::path metadataPath = profiles.profileDir / "metadata.csv";

    profiles.samples = append ? countExistingLdcProfileSamples(stepsPath) : 0;
    if (append && profiles.samples > 0 && !existingLdcProfileSamplesAreCompatible(metadataPath))
    {
        std::cerr << "Existing LDC profile samples in " << profiles.profileDir
                  << " use legacy scaled velocity storage. Refusing to append physical samples into the same files."
                  << std::endl;
        std::cerr << "Post-process the existing data as-is, or move/remove profiles/ before continuing diagnostics."
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const std::ios::openmode sampleMode = std::ios::binary | (append ? std::ios::app : std::ios::trunc);
    profiles.cxSamplesOutput.open(cxPath, sampleMode);
    profiles.cySamplesOutput.open(cyPath, sampleMode);

    const bool writeStepsHeader = !append || profiles.samples == 0 || !std::filesystem::exists(stepsPath);
    profiles.stepsOutput.open(stepsPath, std::ios::out | (append && !writeStepsHeader ? std::ios::app : std::ios::trunc));

    if (!profiles.cxSamplesOutput || !profiles.cySamplesOutput || !profiles.stepsOutput)
    {
        std::cerr << "Could not open LDC profile streaming outputs in " << profiles.profileDir << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (writeStepsHeader)
    {
        profiles.stepsOutput << "sample,step\n";
    }

    writeLdcProfileCoordinates(profiles.profileDir / "centerline_cx_coordinates.csv", NX);
    writeLdcProfileCoordinates(profiles.profileDir / "centerline_cy_coordinates.csv", NZ);
    writeLdcProfileMetadata(profiles.profileDir, profiles.samples);

    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&profiles.deviceSample),
                                 LDC_PROFILE_VALUES * sizeof(real_t));
    if (err != cudaSuccess)
    {
        return err;
    }

    return cudaSuccess;
}

static inline cudaError_t destroyLdcProfileSamples(
    LdcProfileSamples &profiles)
{
    profiles.cxSamplesOutput.close();
    profiles.cySamplesOutput.close();
    profiles.stepsOutput.close();

    cudaError_t err = cudaSuccess;
    if (profiles.deviceSample != nullptr)
    {
        err = cudaFree(profiles.deviceSample);
    }

    profiles.deviceSample = nullptr;
    profiles.hostSample.clear();
    profiles.profileDir.clear();
    profiles.samples = 0;
    return err;
}

static inline void writeLdcProfileSampleChunk(
    std::ofstream &out,
    const std::vector<real_t> &sample,
    const natural_t offset,
    const natural_t length,
    const std::filesystem::path &profileDir)
{
    out.write(reinterpret_cast<const char *>(sample.data() + offset),
              static_cast<std::streamsize>(length * LDC_PROFILE_FIELDS * sizeof(real_t)));
    if (!out)
    {
        std::cerr << "Could not write LDC profile samples in " << profileDir << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline cudaError_t writeLdcProfileSample(
    LdcProfileSamples &profiles,
    const real_t *__restrict__ moments,
    const natural_t step)
{
    constexpr natural_t profileLength = NX > NZ ? NX : NZ;
    constexpr natural_t blocks = (profileLength + LDC_PROFILE_THREADS - 1) / LDC_PROFILE_THREADS;

    writeLdcProfileSampleKernel<<<blocks, LDC_PROFILE_THREADS>>>(moments, profiles.deviceSample, 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return err;
    }

    err = cudaMemcpy(profiles.hostSample.data(),
                     profiles.deviceSample,
                     LDC_PROFILE_VALUES * sizeof(real_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        return err;
    }

    writeLdcProfileSampleChunk(profiles.cxSamplesOutput, profiles.hostSample, LDC_PROFILE_CX_OFFSET, NX, profiles.profileDir);
    writeLdcProfileSampleChunk(profiles.cySamplesOutput, profiles.hostSample, LDC_PROFILE_CY_OFFSET, NZ, profiles.profileDir);

    profiles.stepsOutput << profiles.samples << ',' << step << '\n';
    if (!profiles.stepsOutput)
    {
        std::cerr << "Could not write LDC profile sample steps in " << profiles.profileDir << std::endl;
        std::exit(EXIT_FAILURE);
    }

    ++profiles.samples;
    profiles.cxSamplesOutput.flush();
    profiles.cySamplesOutput.flush();
    profiles.stepsOutput.flush();
    writeLdcProfileMetadata(profiles.profileDir, profiles.samples);

    return cudaSuccess;
}

static inline void writeLdcProfileSamples(
    LdcProfileSamples &profiles,
    const std::filesystem::path &)
{
    profiles.cxSamplesOutput.flush();
    profiles.cySamplesOutput.flush();
    profiles.stepsOutput.flush();
    writeLdcProfileMetadata(profiles.profileDir, profiles.samples);
}
