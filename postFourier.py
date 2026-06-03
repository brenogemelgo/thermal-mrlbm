caseName = "fourier"
runId = "128"
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
    listAvailableSteps,
    readScalarField,
    readPackedField,
    makeCoordinateGrids,
    getInt,
    getFloat,
    writeReport,
    writeDatFile,
    writeGridDatFile,
)


def getWaveNumbers(metadata):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")

    try:
        kx = getFloat(metadata, "KX")
        ky = getFloat(metadata, "KY")
        kz = getFloat(metadata, "KZ")
        k2 = getFloat(metadata, "K2")
    except RuntimeError:
        modeX = getInt(metadata, "MODE_X")
        modeY = getInt(metadata, "MODE_Y")
        modeZ = getInt(metadata, "MODE_Z")
        kx = 2.0 * np.pi * modeX / nx
        ky = 2.0 * np.pi * modeY / ny
        kz = 2.0 * np.pi * modeZ / nz
        k2 = kx * kx + ky * ky + kz * kz

    return kx, ky, kz, k2


def analyticalTemperature(metadata, stepValue, xGrid, yGrid, zGrid):
    t0 = getFloat(metadata, "T0")
    ampT = getFloat(metadata, "AMP_T")
    ux0 = getFloat(metadata, "UX0")
    uy0 = getFloat(metadata, "UY0")
    uz0 = getFloat(metadata, "UZ0")
    alpha = getFloat(metadata, "ALPHA")
    kx, ky, kz, k2 = getWaveNumbers(metadata)

    timeValue = float(stepValue)
    phase = (
        kx * xGrid
        + ky * yGrid
        + kz * zGrid
        - (kx * ux0 + ky * uy0 + kz * uz0) * timeValue
    )
    amplitude = ampT * np.exp(-alpha * k2 * timeValue)
    return t0 + amplitude * np.sin(phase), phase, amplitude


def estimateAmplitude(temperatureValues, phaseValues, t0):
    sampleCount = temperatureValues.size
    centeredValues = temperatureValues - t0
    sinMode = np.sin(phaseValues)
    cosMode = np.cos(phaseValues)
    aSin = 2.0 / sampleCount * np.sum(centeredValues * sinMode)
    aCos = 2.0 / sampleCount * np.sum(centeredValues * cosMode)
    return np.sqrt(aSin * aSin + aCos * aCos)


def savePlots(
    postDir,
    metadata,
    stepValue,
    temperatureNum,
    temperatureAna,
    errorValues,
    phaseValues,
    amplitudeAna,
):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")
    yMid = ny // 2
    zMid = nz // 2

    xValues = np.arange(nx)
    writeDatFile(
        postDir / f"fourierCenterlineStep{stepValue}.dat",
        ["x", "temperatureNumerical", "temperatureAnalytical", "temperatureError"],
        [
            xValues,
            temperatureNum[zMid, yMid, :],
            temperatureAna[zMid, yMid, :],
            errorValues[zMid, yMid, :],
        ],
    )

    plt.figure(figsize=(8, 4))
    plt.plot(xValues, temperatureNum[zMid, yMid, :], label="numerical")
    plt.plot(xValues, temperatureAna[zMid, yMid, :], "--", label="analytical")
    plt.xlabel("x")
    plt.ylabel("temperature")
    plt.title(f"Fourier centerline, step {stepValue}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(postDir / f"fourierCenterlineStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    xGrid2d, yGrid2d = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    writeGridDatFile(
        postDir / f"fourierErrorMidplaneStep{stepValue}.dat",
        ["x", "y", "temperatureError"],
        [xGrid2d, yGrid2d, errorValues[zMid, :, :]],
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(errorValues[zMid, :, :], origin="lower", aspect="auto")
    plt.colorbar(label="temperature error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Fourier mid-plane error, step {stepValue}")
    plt.tight_layout()
    plt.savefig(postDir / f"fourierErrorMidplaneStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def saveAmplitudeDecay(postDir, runDir, metadata, stepValues, xGrid, yGrid, zGrid):
    if len(stepValues) < 2:
        return

    t0 = getFloat(metadata, "T0")
    alpha = getFloat(metadata, "ALPHA")
    ampT = getFloat(metadata, "AMP_T")
    kx, ky, kz, k2 = getWaveNumbers(metadata)
    ux0 = getFloat(metadata, "UX0")
    uy0 = getFloat(metadata, "UY0")
    uz0 = getFloat(metadata, "UZ0")

    amplitudeNumValues = []
    amplitudeAnaValues = []
    for stepValue in stepValues:
        temperatureValues = readPackedField(runDir, metadata, "temperature", stepValue)
        phaseValues = (
            kx * xGrid
            + ky * yGrid
            + kz * zGrid
            - (kx * ux0 + ky * uy0 + kz * uz0) * float(stepValue)
        )
        amplitudeNumValues.append(estimateAmplitude(temperatureValues, phaseValues, t0))
        amplitudeAnaValues.append(ampT * np.exp(-alpha * k2 * float(stepValue)))

    amplitudeNumArray = np.asarray(amplitudeNumValues)
    amplitudeAnaArray = np.asarray(amplitudeAnaValues)
    writeDatFile(
        postDir / "fourierAmplitudeDecay.dat",
        ["step", "amplitudeNumerical", "amplitudeAnalytical", "relativeAmplitudeError"],
        [
            np.asarray(stepValues),
            amplitudeNumArray,
            amplitudeAnaArray,
            np.abs(amplitudeNumArray - amplitudeAnaArray)
            / np.maximum(np.abs(amplitudeAnaArray), 1.0e-30),
        ],
    )

    plt.figure(figsize=(7, 4))
    plt.plot(stepValues, amplitudeNumValues, "o-", label="numerical")
    plt.plot(stepValues, amplitudeAnaValues, "--", label="analytical")
    plt.xlabel("step")
    plt.ylabel("mode amplitude")
    plt.title("Fourier amplitude decay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(postDir / "fourierAmplitudeDecay.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    postDir = getPostDir(runDir)

    temperatureNum, stepValue = readScalarField(
        runDir, metadata, "temperature", selectedStep
    )
    xGrid, yGrid, zGrid = makeCoordinateGrids(metadata)
    temperatureAna, phaseValues, amplitudeAna = analyticalTemperature(
        metadata, stepValue, xGrid, yGrid, zGrid
    )

    errorValues = temperatureNum - temperatureAna
    relativeL2Error = np.linalg.norm(errorValues.ravel()) / max(
        np.linalg.norm((temperatureAna - getFloat(metadata, "T0")).ravel()), 1.0e-30
    )
    absoluteLinfError = np.max(np.abs(errorValues))
    meanError = np.mean(errorValues)
    rmsError = np.sqrt(np.mean(errorValues * errorValues))
    amplitudeNum = estimateAmplitude(
        temperatureNum, phaseValues, getFloat(metadata, "T0")
    )
    relativeAmplitudeError = abs(amplitudeNum - amplitudeAna) / max(
        abs(amplitudeAna), 1.0e-30
    )

    reportLines = [
        f"selectedStep = {stepValue}",
        f"decayRate = {getFloat(metadata, 'DECAY_RATE')}",
        f"analyticalAmplitude = {amplitudeAna:.17g}",
        f"numericalAmplitude = {amplitudeNum:.17g}",
        f"relativeAmplitudeError = {relativeAmplitudeError:.17g}",
        f"relativeL2Error = {relativeL2Error:.17g}",
        f"absoluteLinfError = {absoluteLinfError:.17g}",
        f"meanError = {meanError:.17g}",
        f"rmsError = {rmsError:.17g}",
    ]
    writeReport(postDir / "fourierReport.txt", reportLines)

    savePlots(
        postDir,
        metadata,
        stepValue,
        temperatureNum,
        temperatureAna,
        errorValues,
        phaseValues,
        amplitudeAna,
    )
    saveAmplitudeDecay(
        postDir, runDir, metadata, listAvailableSteps(runDir), xGrid, yGrid, zGrid
    )


main()
