caseName = "jet"
runId = "000"
selectedStep = None
outputRoot = "output"
showPlots = False
figureDpi = 200

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
    tryReadStandaloneScalarField,
    getInt,
    getFloat,
    getBool,
    getMetadataValue,
    writeReport,
)


def saveJetPlots(postDir, metadata, stepValue, thetaValues, uzValues, thetaMeanValues, uzMeanValues):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")
    centerX = nx // 2
    centerY = ny // 2
    zValues = np.arange(nz)

    plt.figure(figsize=(7, 5))
    plt.imshow(thetaValues[:, centerY, :], origin="lower", extent=(0, nx - 1, 0, nz - 1), aspect="auto")
    plt.colorbar(label="theta")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Jet temperature x-z slice, step {stepValue}")
    plt.tight_layout()
    plt.savefig(postDir / f"jetTemperatureXzStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.imshow(uzValues[:, centerY, :], origin="lower", extent=(0, nx - 1, 0, nz - 1), aspect="auto")
    plt.colorbar(label="uz")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Jet uz x-z slice, step {stepValue}")
    plt.tight_layout()
    plt.savefig(postDir / f"jetUzXzStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(zValues, thetaValues[:, centerY, centerX], label="theta")
    plt.plot(zValues, uzValues[:, centerY, centerX], label="uz")

    if thetaMeanValues is not None:
        plt.plot(zValues, thetaMeanValues[:, centerY, centerX], "--", label="thetaMean")
    if uzMeanValues is not None:
        plt.plot(zValues, uzMeanValues[:, centerY, centerX], "--", label="uzMean")

    plt.xlabel("z")
    plt.ylabel("centerline value")
    plt.title(f"Jet centerline profiles, step {stepValue}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(postDir / f"jetCenterlineProfilesStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    postDir = getPostDir(runDir)

    temperatureValues, stepValue = readScalarField(runDir, metadata, "temperature", selectedStep)
    uzValues, stepValue = readScalarField(runDir, metadata, "uz", stepValue)

    tInlet = getFloat(metadata, "T_INLET")
    tAmbient = getFloat(metadata, "T_AMBIENT")
    deltaT = tInlet - tAmbient
    if abs(deltaT) < 1.0e-30:
        raise RuntimeError("Cannot compute theta because T_INLET - T_AMBIENT is zero")

    thetaValues = (temperatureValues - tAmbient) / deltaT
    temperatureMeanValues = tryReadStandaloneScalarField(runDir, metadata, "temperatureMean")
    uzMeanValues = tryReadStandaloneScalarField(runDir, metadata, "uzMean")
    thetaMeanValues = None
    if temperatureMeanValues is not None:
        thetaMeanValues = (temperatureMeanValues - tAmbient) / deltaT

    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")
    centerX = nx // 2
    centerY = ny // 2
    samplePositions = sorted(set([0, nz // 4, nz // 2, (3 * nz) // 4, nz - 1]))
    averagedFound = temperatureMeanValues is not None and uzMeanValues is not None

    reportLines = [
        f"selectedStep = {stepValue}",
        f"RE = {getFloat(metadata, 'RE')}",
        f"PR = {getFloat(metadata, 'PR')}",
        f"RI = {getFloat(metadata, 'RI')}",
        f"U_JET = {getFloat(metadata, 'U_JET')}",
        f"D_JET = {getFloat(metadata, 'D_JET')}",
        f"NU = {getFloat(metadata, 'NU')}",
        f"ALPHA = {getFloat(metadata, 'ALPHA')}",
        f"GBETA = {getFloat(metadata, 'GBETA')}",
        f"minTemperature = {np.min(temperatureValues):.17g}",
        f"maxTemperature = {np.max(temperatureValues):.17g}",
        f"minUz = {np.min(uzValues):.17g}",
        f"maxUz = {np.max(uzValues):.17g}",
        f"timeAveragedFieldsFound = {averagedFound}",
        f"timeAverageEnabled = {getBool(metadata, 'timeAverageEnabled', False)}",
        f"timeAverageCount = {getMetadataValue(metadata, 'timeAverageCount', 0)}",
    ]

    for zValue in samplePositions:
        reportLines.append(
            f"centerline z={zValue}: theta = {thetaValues[zValue, centerY, centerX]:.17g}, uz = {uzValues[zValue, centerY, centerX]:.17g}"
        )

    writeReport(postDir / "jetReport.txt", reportLines)
    saveJetPlots(postDir, metadata, stepValue, thetaValues, uzValues, thetaMeanValues, uzMeanValues)


main()
