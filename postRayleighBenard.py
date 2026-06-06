caseName = "rayleigh_benard"
runId = "000"
selectedStep = None
outputRoot = "output"
showPlots = False
figureDpi = 600

import numpy as np
import matplotlib

if not showPlots:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from postCommon import (
    getRunDir,
    readMetadata,
    getPostDir,
    readScalarField,
    getInt,
    getFloat,
    getBool,
    writeReport,
    writeDatFile,
    writeGridDatFile,
)


def tryReadField(runDir, metadata, fieldName, stepValue):
    try:
        fieldValues, _ = readScalarField(runDir, metadata, fieldName, stepValue)
        return fieldValues
    except RuntimeError:
        return None


def conductiveProfile(metadata):
    nz = getInt(metadata, "NZ")
    hValue = getFloat(metadata, "H")
    tHot = getFloat(metadata, "THOT")
    deltaT = getFloat(metadata, "DELTA_T")
    zIndex = np.arange(nz, dtype=np.float64)
    zCoord = zIndex / hValue
    temperatureExact = tHot - deltaT * zCoord
    return zIndex, zCoord, temperatureExact


def wallDerivativeBottom(meanTemperature):
    if meanTemperature.size >= 3:
        return (-3.0 * meanTemperature[0] + 4.0 * meanTemperature[1] - meanTemperature[2]) * 0.5
    return meanTemperature[1] - meanTemperature[0]


def wallDerivativeTop(meanTemperature):
    if meanTemperature.size >= 3:
        return (3.0 * meanTemperature[-1] - 4.0 * meanTemperature[-2] + meanTemperature[-3]) * 0.5
    return meanTemperature[-1] - meanTemperature[-2]


def saveSlicePlots(postDir, metadata, stepValue, temperatureValues, uxValues, uzValues):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")
    yMid = ny // 2
    xGrid2d, zGrid2d = np.meshgrid(np.arange(nx), np.arange(nz), indexing="xy")

    temperatureSlice = temperatureValues[:, yMid, :]
    writeGridDatFile(
        postDir / f"rayleighBenardTemperatureXzStep{stepValue}.dat",
        ["x", "z", "temperature"],
        [xGrid2d, zGrid2d, temperatureSlice],
    )
    plt.figure(figsize=(8, 4))
    plt.imshow(
        temperatureSlice,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        aspect="auto",
    )
    plt.colorbar(label="temperature")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Rayleigh-Benard temperature, y={yMid}, step {stepValue}")
    plt.tight_layout()
    plt.savefig(postDir / f"rayleighBenardTemperatureXzStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    if uzValues is None:
        return

    uzSlice = uzValues[:, yMid, :]
    writeGridDatFile(
        postDir / f"rayleighBenardUzXzStep{stepValue}.dat",
        ["x", "z", "uz"],
        [xGrid2d, zGrid2d, uzSlice],
    )
    plt.figure(figsize=(8, 4))
    plt.imshow(
        uzSlice,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        aspect="auto",
    )
    plt.colorbar(label="uz")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Rayleigh-Benard uz, y={yMid}, step {stepValue}")
    plt.tight_layout()
    plt.savefig(postDir / f"rayleighBenardUzXzStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    if uxValues is None:
        return

    uxSlice = uxValues[:, yMid, :]
    dUxDz = np.gradient(uxSlice, axis=0)
    dUzDx = np.gradient(uzSlice, axis=1)
    omegaY = dUxDz - dUzDx
    writeGridDatFile(
        postDir / f"rayleighBenardOmegaYStep{stepValue}.dat",
        ["x", "z", "omegaY"],
        [xGrid2d, zGrid2d, omegaY],
    )
    plt.figure(figsize=(8, 4))
    plt.imshow(
        omegaY,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        aspect="auto",
        cmap="coolwarm",
    )
    plt.colorbar(label="omega_y")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Rayleigh-Benard omega_y, y={yMid}, step {stepValue}")
    plt.tight_layout()
    plt.savefig(postDir / f"rayleighBenardOmegaYStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def saveProfilePlot(postDir, stepValue, zCoord, meanTemperature, temperatureExact):
    writeDatFile(
        postDir / f"rayleighBenardMeanTemperatureStep{stepValue}.dat",
        ["Z", "meanTemperature", "conductiveTemperature", "temperatureDifference"],
        [zCoord, meanTemperature, temperatureExact, meanTemperature - temperatureExact],
    )
    plt.figure(figsize=(5, 6))
    plt.plot(meanTemperature, zCoord, label="simulation")
    plt.plot(temperatureExact, zCoord, "--", label="conduction")
    plt.xlabel("temperature")
    plt.ylabel("Z")
    plt.title(f"Mean temperature profile, step {stepValue}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(postDir / f"rayleighBenardMeanTemperatureStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    postDir = getPostDir(runDir)

    temperatureValues, stepValue = readScalarField(
        runDir, metadata, "temperature", selectedStep
    )
    uxValues = tryReadField(runDir, metadata, "ux", stepValue)
    uyValues = tryReadField(runDir, metadata, "uy", stepValue)
    uzValues = tryReadField(runDir, metadata, "uz", stepValue)
    rhoValues = tryReadField(runDir, metadata, "rho", stepValue)

    zIndex, zCoord, temperatureExact = conductiveProfile(metadata)
    meanTemperature = np.mean(temperatureValues, axis=(1, 2))
    conductiveField = temperatureExact[:, None, None]
    temperatureError = temperatureValues - conductiveField
    l2Temperature = np.sqrt(np.mean(temperatureError * temperatureError))
    linfTemperature = np.max(np.abs(temperatureError))

    hValue = getFloat(metadata, "H")
    deltaT = getFloat(metadata, "DELTA_T")
    dTdzBottom = wallDerivativeBottom(meanTemperature)
    dTdzTop = wallDerivativeTop(meanTemperature)
    nuBottom = -hValue / deltaT * dTdzBottom
    nuTop = -hValue / deltaT * dTdzTop
    nuAverage = 0.5 * (nuBottom + nuTop)

    if uxValues is not None and uyValues is not None and uzValues is not None:
        velocityMagnitudeSquared = uxValues * uxValues + uyValues * uyValues + uzValues * uzValues
        velocityMagnitude = np.sqrt(velocityMagnitudeSquared)
        velocityMax = np.max(velocityMagnitude)
        kineticEnergy = 0.5 * np.mean(velocityMagnitudeSquared)
        raValue = getFloat(metadata, "RA")
        prValue = getFloat(metadata, "PR")
        thetaFluctuation = temperatureError / deltaT
        nuVolume = 1.0 + np.sqrt(max(raValue * prValue, 0.0)) * np.mean(uzValues * thetaFluctuation)
    else:
        velocityMax = np.nan
        kineticEnergy = np.nan
        nuVolume = np.nan

    reportLines = [
        f"selectedStep = {stepValue}",
        f"RA = {getFloat(metadata, 'RA')}",
        f"PR = {getFloat(metadata, 'PR')}",
        f"NU = {getFloat(metadata, 'NU')}",
        f"ALPHA = {getFloat(metadata, 'ALPHA')}",
        f"GBETA = {getFloat(metadata, 'GBETA')}",
        f"H = {hValue}",
        f"ASPECT_RATIO = {getFloat(metadata, 'ASPECT_RATIO')}",
        f"CONDUCTION_ONLY = {getBool(metadata, 'CONDUCTION_ONLY', False)}",
        f"PERTURBATION_AMPLITUDE = {getFloat(metadata, 'PERTURBATION_AMPLITUDE', 0.0)}",
        f"temperatureL2Error = {l2Temperature:.17g}",
        f"temperatureLinfError = {linfTemperature:.17g}",
        f"velocityMax = {velocityMax:.17g}",
        f"kineticEnergy = {kineticEnergy:.17g}",
        f"nuBottom = {nuBottom:.17g}",
        f"nuTop = {nuTop:.17g}",
        f"nuAverage = {nuAverage:.17g}",
        f"nuVolumeOptional = {nuVolume:.17g}",
        "nuVolumeOptional uses theta=(T-T_conductive)/DeltaT and is secondary to wall-gradient Nu.",
        f"temperatureMin = {np.min(temperatureValues):.17g}",
        f"temperatureMax = {np.max(temperatureValues):.17g}",
        f"meanRho = {np.mean(rhoValues):.17g}" if rhoValues is not None else "meanRho = unavailable",
    ]

    writeReport(postDir / "rayleighBenardReport.txt", reportLines)
    for lineText in reportLines:
        print(lineText)

    saveProfilePlot(postDir, stepValue, zCoord, meanTemperature, temperatureExact)
    saveSlicePlots(postDir, metadata, stepValue, temperatureValues, uxValues, uzValues)

    writeDatFile(
        postDir / f"rayleighBenardSummaryStep{stepValue}.dat",
        [
            "step",
            "temperatureL2Error",
            "temperatureLinfError",
            "velocityMax",
            "kineticEnergy",
            "nuBottom",
            "nuTop",
            "nuAverage",
            "nuVolumeOptional",
        ],
        [
            np.array([stepValue]),
            np.array([l2Temperature]),
            np.array([linfTemperature]),
            np.array([velocityMax]),
            np.array([kineticEnergy]),
            np.array([nuBottom]),
            np.array([nuTop]),
            np.array([nuAverage]),
            np.array([nuVolume]),
        ],
    )


main()
