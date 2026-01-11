import struct
import numpy as np
import png
import re
import sys
import csv
from PIL import Image
import h5py


FLO_TAG_FLOAT = (
    202021.25  # first 4 bytes in flo file; check for this when READING the file
)
FLO_TAG_STRING = "PIEH"  # first 4 bytes in flo file; use this when WRITING the file
FLO_UNKNOWN_FLOW_THRESH = 1e9  # flo format threshold for unknown values
FLO_UNKNOWN_FLOW = 1e10  # value to use to represent unknown flow in flo file format


def readFlowFile(filepath):
    """read flow files in several formats. The resulting flow has shape height x width x 2.
    For positions where there is no groundtruth available, the flow is set to np.nan.
    Supports flo (Sintel), png (KITTI), npy (numpy), pfm (FlyingThings3D) and flo5 (Spring) file format.
    filepath: path to the flow file
    returns: flow with shape height x width x 2
    """
    if filepath.endswith(".flo"):
        return readFloFlow(filepath)
    elif filepath.endswith(".png"):
        return readPngFlow(filepath)
    elif filepath.endswith(".npy"):
        return readNpyFlow(filepath)
    elif filepath.endswith(".pfm"):
        return readPfmFlow(filepath)
    elif filepath.endswith(".flo5"):
        return readFlo5Flow(filepath)
    else:
        raise ValueError(f"readFlowFile: Unknown file format for {filepath}")


def writeFlowFile(flow, filepath):
    """write optical flow to file. Supports flo (Sintel), png (KITTI) and npy (numpy) file format.
    flow: optical flow with shape height x width x 2. Invalid values should be represented as np.nan
    filepath: file path where to write the flow
    """
    if not filepath:
        raise ValueError("writeFlowFile: empty filepath")

    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise IOError(
            f"writeFlowFile {filepath}: expected shape height x width x 2 but received {flow.shape}"
        )

    if flow.shape[0] > flow.shape[1]:
        print(
            f"write flo file {filepath}: Warning: Are you writing an upright image? Expected shape height x width x 2, got {flow.shape}"
        )

    if filepath.endswith(".flo"):
        return writeFloFlow(flow, filepath)
    elif filepath.endswith(".png"):
        return writePngFlow(flow, filepath)
    elif filepath.endswith(".npy"):
        return writeNpyFile(flow, filepath)
    elif filepath.endswith(".flo5"):
        return writeFlo5File(flow, filepath)
    else:
        raise ValueError(f"writeFlowFile: Unknown file format for {filepath}")


def readFloFlow(filepath):
    """read optical flow from file stored in .flo file format as used in the Sintel dataset (Butler et al., 2012)
    filepath: path to file where to read from
    returns: flow as a numpy array with shape height x width x 2
    ---
    ".flo" file format used for optical flow evaluation

    Stores 2-band float image for horizontal (u) and vertical (v) flow components.
    Floats are stored in little-endian order.
    A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.

    bytes  contents

    0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
            (just a sanity check that floats are represented correctly)
    4-7     width as an integer
    8-11    height as an integer
    12-end  data (width*height*2*4 bytes total)
            the float values for u and v, interleaved, in row order, i.e.,
            u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
    """
    if filepath is None:
        raise IOError("read flo file: empty filename")

    if not filepath.endswith(".flo"):
        raise IOError(f"read flo file ({filepath}): extension .flo expected")

    with open(filepath, "rb") as stream:
        tag = struct.unpack("f", stream.read(4))[0]
        width = struct.unpack("i", stream.read(4))[0]
        height = struct.unpack("i", stream.read(4))[0]

        if tag != FLO_TAG_FLOAT:  # simple test for correct endian-ness
            raise IOError(
                f"read flo file({filepath}): wrong tag (possibly due to big-endian machine?)"
            )

        # another sanity check to see that integers were read correctly (99999 should do the trick...)
        if width < 1 or width > 99999:
            raise IOError(f"read flo file({filepath}): illegal width {width}")

        if height < 1 or height > 99999:
            raise IOError(f"read flo file({filepath}): illegal height {height}")

        nBands = 2
        flow = []

        n = nBands * width
        for _ in range(height):
            data = stream.read(n * 4)
            if data is None:
                raise IOError(f"read flo file({filepath}): file is too short")
            data = np.asarray(struct.unpack(f"{n}f", data))
            data = data.reshape((width, nBands))
            flow.append(data)

        if stream.read(1) != b"":
            raise IOError(f"read flo file({filepath}): file is too long")

        flow = np.asarray(flow)
        # unknown values are set to nan
        flow[np.abs(flow) > FLO_UNKNOWN_FLOW_THRESH] = np.nan

        return flow


def writeFloFlow(flow, filepath):
    """
    write optical flow in .flo format to file as used in the Sintel dataset (Butler et al., 2012)
    flow: optical flow with shape height x width x 2
    filepath: optical flow file path to be saved
    ---
    ".flo" file format used for optical flow evaluation

    Stores 2-band float image for horizontal (u) and vertical (v) flow components.
    Floats are stored in little-endian order.
    A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.

    bytes  contents

    0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
            (just a sanity check that floats are represented correctly)
    4-7     width as an integer
    8-11    height as an integer
    12-end  data (width*height*2*4 bytes total)
            the float values for u and v, interleaved, in row order, i.e.,
            u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
    """

    height, width, nBands = flow.shape

    with open(filepath, "wb") as f:
        if f is None:
            raise IOError(f"write flo file {filepath}: file could not be opened")

        # write header
        result = f.write(FLO_TAG_STRING.encode("ascii"))
        result += f.write(struct.pack("i", width))
        result += f.write(struct.pack("i", height))
        if result != 12:
            raise IOError(f"write flo file {filepath}: problem writing header")

        # write content
        n = nBands * width
        for i in range(height):
            data = flow[i, :, :].flatten()
            data[np.isnan(data)] = FLO_UNKNOWN_FLOW
            result = f.write(struct.pack(f"{n}f", *data))
            if result != n * 4:
                raise IOError(f"write flo file {filepath}: problem writing row {i}")


def readPngFlow(filepath):
    """read optical flow from file stored in png file format as used in the KITTI 12 (Geiger et al., 2012) and KITTI 15 (Menze et al., 2015) dataset.
    filepath: path to file where to read from
    returns: flow as a numpy array with shape height x width x 2. Invalid values are represented as np.nan
    """
    # adapted from https://github.com/liruoteng/OpticalFlowToolkit
    flow_object = png.Reader(filename=filepath)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]["size"]
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = flow[:, :, 2] == 0
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2**15) / 64.0
    flow[invalid_idx, 0] = np.nan
    flow[invalid_idx, 1] = np.nan
    return flow[:, :, :2]


def writePngFlow(flow, filename):
    """write optical flow to file png file format as used in the KITTI 12 (Geiger et al., 2012) and KITTI 15 (Menze et al., 2015) dataset.
    flow: optical flow in shape height x width x 2, invalid values should be represented as np.nan
    filepath: path to file where to write to
    """
    flow = 64.0 * flow + 2**15
    width = flow.shape[1]
    height = flow.shape[0]
    valid_map = np.ones([flow.shape[0], flow.shape[1], 1])
    valid_map[np.isnan(flow[:, :, 0]) | np.isnan(flow[:, :, 1])] = 0
    flow = np.nan_to_num(flow)
    flow = np.concatenate([flow, valid_map], axis=-1)
    flow = np.clip(flow, 0, 2**16 - 1)
    flow = flow.astype(np.uint16)
    flow = np.reshape(flow, (-1, width * 3))
    with open(filename, "wb") as f:
        writer = png.Writer(width=width, height=height, bitdepth=16, greyscale=False)
        writer.write(f, flow)


def readNpyFlow(filepath):
    """read numpy array from file.
    filepath: file to read from
    returns: numpy array
    """
    return np.load(filepath)


def writeNpyFile(arr, filepath):
    """write numpy array to file.
    arr: numpy array to write
    filepath: file to write to
    """
    np.save(filepath, arr)


def writeFlo5File(flow, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)


def readFlo5Flow(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(
                f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?"
            )
        return f["flow"][()]


def readPfmFlow(filepath):
    """read optical flow from file stored in pfm file format as used in the FlyingThings3D (Mayer et al., 2016) dataset.
    filepath: path to file where to read from
    returns: flow as a numpy array with shape height x width x 2.
    """
    flow = readPfmFile(filepath)
    if len(flow.shape) != 3:
        raise IOError(
            f"read pfm flow: PFM file has wrong shape (assumed to be w x h x 3): {flow.shape}"
        )
    if flow.shape[2] != 3:
        raise IOError(
            f"read pfm flow: PFM file has wrong shape (assumed to be w x h x 3): {flow.shape}"
        )
    # remove third channel -> is all zeros
    return flow[:, :, :2]


def readPfmFile(filepath):
    """
    adapted from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    """
    file = open(filepath, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == "PF":
        color = True
    elif header.decode("ascii") == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data  # , scale


def writePfmFile(image, filepath):
    """
    adapted from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    """
    scale = 1
    file = open(filepath, "wb")

    color = None

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif (
        len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
    ):  # greyscale
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    file.write("PF\n" if color else "Pf\n".encode())
    file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    file.write("%f\n".encode() % scale)

    image.tofile(file)


def readDispFile(filepath):
    """read disparity (or disparity change) from file. The resulting numpy array has shape height x width.
    For positions where there is no groundtruth available, the value is set to np.nan.
    Supports png (KITTI), npy (numpy) and pfm (FlyingThings3D) file format.
    filepath: path to the flow file
    returns: disparity with shape height x width
    """
    if filepath.endswith(".png"):
        return readPngDisp(filepath)
    elif filepath.endswith(".npy"):
        return readNpyFlow(filepath)
    elif filepath.endswith(".pfm"):
        return readPfmDisp(filepath)
    elif filepath.endswith(".dsp5"):
        return readDsp5Disp(filepath)
    else:
        raise ValueError(f"readDispFile: Unknown file format for {filepath}")


def readPngDisp(filepath):
    """read disparity from file stored in png file format as used in the KITTI 12 (Geiger et al., 2012) and KITTI 15 (Menze et al., 2015) dataset.
    filepath: path to file where to read from
    returns: disparity as a numpy array with shape height x width. Invalid values are represented as np.nan
    """
    # adapted from https://github.com/liruoteng/OpticalFlowToolkit
    image_object = png.Reader(filename=filepath)
    image_direct = image_object.asDirect()
    image_data = list(image_direct[2])
    (w, h) = image_direct[3]["size"]
    channel = len(image_data[0]) // w
    if channel != 1:
        raise IOError("read png disp: assumed channels to be 1!")
    disp = np.zeros((h, w), dtype=np.float64)
    for i in range(len(image_data)):
        disp[i, :] = image_data[i][:]
    disp[disp == 0] = np.nan
    return disp[:, :] / 256.0


def readPfmDisp(filepath):
    """read disparity or disparity change from file stored in pfm file format as used in the FlyingThings3D (Mayer et al., 2016) dataset.
    filepath: path to file where to read from
    returns: disparity as a numpy array with shape height x width. Invalid values are represented as np.nan
    """
    disp = readPfmFile(filepath)
    if len(disp.shape) != 2:
        raise IOError(
            f"read pfm disp: PFM file has wrong shape (assumed to be w x h): {disp.shape}"
        )
    return disp


def writePngDisp(disp, filepath):
    """write disparity to png file format as used in the KITTI 12 (Geiger et al., 2012) and KITTI 15 (Menze et al., 2015) dataset.
    disp: disparity in shape height x width, invalid values should be represented as np.nan
    filepath: path to file where to write to
    """
    disp = 256 * disp
    width = disp.shape[1]
    height = disp.shape[0]
    disp = np.clip(disp, 0, 2**16 - 1)
    disp = np.nan_to_num(disp).astype(np.uint16)
    disp = np.reshape(disp, (-1, width))
    with open(filepath, "wb") as f:
        writer = png.Writer(width=width, height=height, bitdepth=16, greyscale=True)
        writer.write(f, disp)


def writeDsp5File(disp, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("disparity", data=disp, compression="gzip", compression_opts=5)


def readDsp5Disp(filename):
    with h5py.File(filename, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(
                f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?"
            )
        return f["disparity"][()]


def writeDispFile(disp, filepath):
    """write disparity to file. Supports png (KITTI) and npy (numpy) file format.
    disp: disparity with shape height x width. Invalid values should be represented as np.nan
    filepath: file path where to write the flow
    """
    if not filepath:
        raise ValueError("writeDispFile: empty filepath")

    if len(disp.shape) != 2:
        raise IOError(
            f"writeDispFile {filepath}: expected shape height x width but received {disp.shape}"
        )

    if disp.shape[0] > disp.shape[1]:
        print(
            f"writeDispFile {filepath}: Warning: Are you writing an upright image? Expected shape height x width, got {disp.shape}"
        )

    if filepath.endswith(".png"):
        writePngDisp(disp, filepath)
    elif filepath.endswith(".npy"):
        writeNpyFile(disp, filepath)
    elif filepath.endswith(".dsp5"):
        writeDsp5File(disp, filepath)


def readKITTIObjMap(filepath):
    assert filepath.endswith(".png")
    return np.asarray(Image.open(filepath)) > 0


def readKITTIIntrinsics(filepath, image=2):
    assert filepath.endswith(".txt")

    with open(filepath) as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if row[0] == f"K_{image:02d}:":
                K = np.array(row[1:], dtype=np.float32).reshape(3, 3)
                kvec = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                return kvec


def writePngMapFile(map_, filename):
    Image.fromarray(map_).save(filename)
