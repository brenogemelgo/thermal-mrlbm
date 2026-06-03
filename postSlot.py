caseName = "slot"
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
    getInt,
    getFloat,
    writeReport,
    writeDatFile,
)


def analyticalSlotProfiles(metadata):
    nx = getInt(metadata, "NX")
    lSlot = getFloat(metadata, "L_SLOT")
    tHot = getFloat(metadata, "THOT")
    deltaT = getFloat(metadata, "DELTA_T")
    slotVelocityScale = getFloat(metadata, "slotVelocityScale")

    xIndex = np.arange(nx)
    xCoord = xIndex / lSlot
    temperatureAna = tHot - deltaT * xCoord
    uyAna = slotVelocityScale * xCoord * (1.0 - xCoord) * (1.0 - 2.0 * xCoord) / 12.0
    return xIndex, xCoord, temperatureAna, uyAna


def relativeL2Error(numericalValues, analyticalValues):
    errorValues = numericalValues - analyticalValues
    denominator = max(np.linalg.norm(analyticalValues), 1.0e-30)
    return np.linalg.norm(errorValues) / denominator


def saveSlotPlots(postDir, stepValue, xCoord, temperatureProfile, temperatureAna, uyProfile, uyAna):
    temperatureError = temperatureProfile - temperatureAna
    uyError = uyProfile - uyAna

    writeDatFile(
        postDir / f"slotTemperatureProfileStep{stepValue}.dat",
        ["X", "temperatureNumerical", "temperatureAnalytical"],
        [xCoord, temperatureProfile, temperatureAna],
    )
    plt.figure(figsize=(7, 4))
    plt.plot(xCoord, temperatureProfile, "o-", label="numerical")
    plt.plot(xCoord, temperatureAna, "--", label="analytical")
    plt.xlabel("X")
    plt.ylabel("temperature")
    plt.title(f"Slot temperature profile, step {stepValue}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(postDir / f"slotTemperatureProfileStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    writeDatFile(
        postDir / f"slotUyProfileStep{stepValue}.dat",
        ["X", "uyNumerical", "uyAnalytical"],
        [xCoord, uyProfile, uyAna],
    )
    plt.figure(figsize=(7, 4))
    plt.plot(xCoord, uyProfile, "o-", label="numerical")
    plt.plot(xCoord, uyAna, "--", label="analytical")
    plt.xlabel("X")
    plt.ylabel("uy")
    plt.title(f"Slot vertical velocity profile, step {stepValue}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(postDir / f"slotUyProfileStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    writeDatFile(
        postDir / f"slotErrorProfilesStep{stepValue}.dat",
        ["X", "temperatureError", "uyError"],
        [xCoord, temperatureError, uyError],
    )
    plt.figure(figsize=(7, 4))
    plt.plot(xCoord, temperatureError, label="T numerical - analytical")
    plt.plot(xCoord, uyError, label="uy numerical - analytical")
    plt.xlabel("X")
    plt.ylabel("error")
    plt.title(f"Slot profile errors, step {stepValue}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(postDir / f"slotErrorProfilesStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    postDir = getPostDir(runDir)

    temperatureValues, stepValue = readScalarField(runDir, metadata, "temperature", selectedStep)
    uyValues, stepValue = readScalarField(runDir, metadata, "uy", stepValue)

    xIndex, xCoord, temperatureAna, uyAna = analyticalSlotProfiles(metadata)
    temperatureProfile = np.mean(temperatureValues, axis=(0, 1))
    uyProfile = np.mean(uyValues, axis=(0, 1))

    temperatureError = temperatureProfile - temperatureAna
    uyError = uyProfile - uyAna
    analyticalMaxAbsUy = getFloat(metadata, "slotVelocityScale") * np.sqrt(3.0) / 216.0
    numericalMaxAbsUy = np.max(np.abs(uyProfile))
    relativeMaxUyError = abs(numericalMaxAbsUy - analyticalMaxAbsUy) / max(abs(analyticalMaxAbsUy), 1.0e-30)
    netFlowRate = np.sum(
        (uyProfile[:-1] + uyProfile[1:]) * (xCoord[1:] - xCoord[:-1]) * 0.5
    )
    temperatureStdYZ = np.max(np.std(temperatureValues, axis=(0, 1)))
    uyStdYZ = np.max(np.std(uyValues, axis=(0, 1)))

    writeDatFile(
        postDir / f"slotProfilesStep{stepValue}.txt",
        [
            "xIndex",
            "X",
            "temperatureNumerical",
            "temperatureAnalytical",
            "uyNumerical",
            "uyAnalytical",
            "temperatureError",
            "uyError",
        ],
        [xIndex, xCoord, temperatureProfile, temperatureAna, uyProfile, uyAna, temperatureError, uyError],
    )

    reportLines = [
        f"selectedStep = {stepValue}",
        f"NU = {getFloat(metadata, 'NU')}",
        f"ALPHA = {getFloat(metadata, 'ALPHA')}",
        f"PR = {getFloat(metadata, 'PR')}",
        f"GBETA = {getFloat(metadata, 'GBETA')}",
        f"L_SLOT = {getFloat(metadata, 'L_SLOT')}",
        f"targetUyMax = {getFloat(metadata, 'targetUyMax')}",
        f"slotVelocityScale = {getFloat(metadata, 'slotVelocityScale')}",
        f"analyticalMaxAbsUy = {analyticalMaxAbsUy:.17g}",
        f"numericalMaxAbsUy = {numericalMaxAbsUy:.17g}",
        f"relativeMaxUyError = {relativeMaxUyError:.17g}",
        f"temperatureRelativeL2Error = {relativeL2Error(temperatureProfile, temperatureAna):.17g}",
        f"uyRelativeL2Error = {relativeL2Error(uyProfile, uyAna):.17g}",
        f"temperatureLinfError = {np.max(np.abs(temperatureError)):.17g}",
        f"uyLinfError = {np.max(np.abs(uyError)):.17g}",
        f"uyMaximumAbsoluteError = {np.max(np.abs(uyError)):.17g}",
        f"netFlowRate = {netFlowRate:.17g}",
        f"temperatureStdYZ = {temperatureStdYZ:.17g}",
        f"uyStdYZ = {uyStdYZ:.17g}",
    ]
    writeReport(postDir / "slotReport.txt", reportLines)
    saveSlotPlots(postDir, stepValue, xCoord, temperatureProfile, temperatureAna, uyProfile, uyAna)


main()
